import math
import transformers
import torch
from .custom.custom_cache import DynamicCacheWithQuery

class HeadDetector():

    def __init__(self, llm_name) -> None:
        # set up LLM
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(llm_name)

        if 'granite' in llm_name.lower():
            from .custom.modeling_granite_attn import GraniteForCausalLM
            BaseLLMClass = GraniteForCausalLM
        elif 'llama' in llm_name.lower():
            from .custom.modeling_llama_attn import LlamaForCausalLM
            BaseLLMClass = LlamaForCausalLM
        elif 'mistral' in llm_name.lower():
            from .custom.modeling_mistral_attn import MistralForCausalLM
            BaseLLMClass = MistralForCausalLM
        elif 'phi' in llm_name.lower():
            from .custom.modeling_phi_attn import Phi3ForCausalLM
            BaseLLMClass = Phi3ForCausalLM
        else:
            print(f'base model {llm_name} not supported')

        self.llm = BaseLLMClass.from_pretrained(
            llm_name,
            torch_dtype=torch.float16,
            attn_implementation='flash_attention_2',
            device_map='cuda'
        )

        # setup prompts
        self.offset = 0
        if 'granite' in llm_name.lower():
            self.prompt_prefix = '<|start_of_role|>user<|end_of_role|>'
            self.prompt_suffix = '<|end_of_text|><|start_of_role|>assistant<|end_of_role|>'
        elif 'llama' in llm_name.lower():
            self.prompt_prefix = '<|start_header_id|>user<|end_header_id|>'
            self.prompt_suffix = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        elif 'mistral' in llm_name.lower():
            self.prompt_prefix = '[INST]'
            self.prompt_suffix = '[/INST]'
            self.offset = 1
        elif 'phi' in llm_name.lower():
            self.prompt_prefix = '<|im_start|>user<|im_sep|>'
            self.prompt_suffix = '<|im_end|><|im_start|>assistant<|im_sep|>'
        self.retrieval_instruction = ' Here are some paragraphs:\n\n'
        self.retrieval_instruction_late = 'Please find information that are relevant to the following query in the paragraphs above.\n\nQuery: '

        # layer info
        self.num_query = 0
        self.num_layer = self.llm.config.num_hidden_layers
        self.num_head = self.llm.config.num_attention_heads
        self.head_score = {}
        for layer in range(self.num_layer):
            for head in range(self.num_head):
                self.head_score[f"{layer}-{head}"] = 0

    def get_head_score(self):
        for head in self.head_score.keys():
            self.head_score[head] /= self.num_query
        return self.head_score

    def compute_retrieval_score(self, query, documents, pos_idx, neg_idx):
        prompt, pos_span, query_span = self.prepare_input(query, documents, pos_idx, neg_idx)
        score = self.score_documents(prompt, pos_span, query_span)
        for layer in range(self.num_layer):
            for head in range(self.num_head):
                self.head_score[f"{layer}-{head}"] += score[layer, head].item()
        self.num_query += 1

    def score_documents(self, prompt, pos_span, query_span):
        tokenized_input = self.tokenizer(prompt,return_tensors='pt').to(self.llm.device)
        _input_ids = tokenized_input.input_ids
        _query_indices = list(range(query_span[0], query_span[1]+1))
        kv_cache=DynamicCacheWithQuery(query_indices=_query_indices)

        with torch.no_grad():
            output = self.llm(
                input_ids=_input_ids,
                use_cache=True,
                past_key_values=kv_cache,
                output_attentions=True
                )
        kv_cache = output.past_key_values

        # loop through all layers and compute attention scores
        all_key_cache = []
        all_query_cache = []
        for i in range(self.num_layer):
            all_key_cache.append(kv_cache.key_cache[i][:,:,:query_span[1]+1])
            all_query_cache.append(kv_cache.query_cache[i])
        all_key_cache = torch.stack(all_key_cache)
        all_query_cache = torch.stack(all_query_cache)

        attn_weights = self._get_attn_weights(all_key_cache, all_query_cache).to('cuda').squeeze(1)
        attn_weights = attn_weights.mean(-2)

        # compute contrastive score
        pos_score = attn_weights[:,:,pos_span[0]:pos_span[1]].sum(-1)
        head_scores = pos_score.to('cpu')

        return head_scores

    def prepare_input(self, query, documents, pos_idx, neg_idx):
        llm_prompt = self.prompt_prefix + self.retrieval_instruction

        for i, doc in enumerate(documents):
            llm_prompt += f'[document {i+1}]'
            start_len = len(self.tokenizer(llm_prompt).input_ids)

            llm_prompt += ' ' + doc
            end_len = len(self.tokenizer(llm_prompt).input_ids) - self.offset

            if i == pos_idx:
                pos_span = (start_len, end_len)
            llm_prompt += '\n\n'

        start_len = len(self.tokenizer(llm_prompt).input_ids)

        llm_prompt += self.retrieval_instruction_late + f'{query.strip()}'
        end_len = len(self.tokenizer(llm_prompt).input_ids) - self.offset
        llm_prompt += self.prompt_suffix

        query_span = (start_len, end_len)

        return llm_prompt, pos_span, query_span

    @classmethod
    def _get_attn_weights(cls, key_states, query_states):
        num_layer, bsz, num_heads, q_len, head_dim = query_states.size()
        num_key_value_heads = key_states.size(2)
        num_key_value_groups = num_heads // num_key_value_heads
        kv_seq_len = key_states.size(-2)

        key_states = key_states.unsqueeze(3).expand(num_layer, bsz, num_key_value_heads, num_key_value_groups, kv_seq_len, head_dim)
        key_states = key_states.reshape(num_layer, bsz, num_heads, kv_seq_len, head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2,-1)) / math.sqrt(head_dim)

        causal_mask = cls._get_causal_mask(attn_weights).to(attn_weights.device)
        attn_weights += causal_mask.unsqueeze(1)
        attn_lses = torch.logsumexp(attn_weights, dim=-1, keepdim=True)
        attn_weights = torch.exp(attn_weights - attn_lses)

        return attn_weights

    @classmethod
    def _get_causal_mask(cls, attn_weights):
        query_len, seq_len = attn_weights.size(-2), attn_weights.size(-1)
        causal_mask = torch.ones_like(attn_weights.transpose(-1,-2).squeeze(1))
        causal_mask = torch.triu(causal_mask, diagonal=-(seq_len-query_len))
        causal_mask = causal_mask.transpose(-1,-2)
        causal_mask = (1-causal_mask) * torch.finfo(causal_mask.dtype).min
        return causal_mask
