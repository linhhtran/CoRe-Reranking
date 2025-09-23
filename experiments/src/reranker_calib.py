import math
import transformers
import torch
from .custom.custom_cache import DynamicCacheWithQuery

class Reranker():

    def __init__(self, llm_name, head_set=None, prune=0.0) -> None:

        # set up the base LLM
        tokenizer = transformers.AutoTokenizer.from_pretrained(llm_name)
        self.tokenizer = tokenizer
        config = transformers.AutoConfig.from_pretrained(llm_name)
        config.num_hidden_layers = int(config.num_hidden_layers * (1-prune)) # prune layers

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

        llm = BaseLLMClass.from_pretrained(
                llm_name,
                config=config,
                torch_dtype=torch.float16,
                attn_implementation='flash_attention_2',
                device_map='cuda'
            )
        self.llm = llm

        # setup prompts
        self.off_set = 0
        if 'granite' in llm_name.lower():
            self.prompt_prefix = '<|start_of_role|>user<|end_of_role|>'
            self.prompt_suffix = '<|end_of_text|><|start_of_role|>assistant<|end_of_role|>'
        elif 'llama' in llm_name.lower():
            self.prompt_prefix = '<|start_header_id|>user<|end_header_id|>'
            self.prompt_suffix = '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        elif 'mistral' in llm_name.lower():
            self.prompt_prefix = '[INST]'
            self.prompt_suffix = '[/INST]'
            self.off_set = 1
        elif 'phi' in llm_name.lower():
            self.prompt_prefix = '<|im_start|>user<|im_sep|>'
            self.prompt_suffix = '<|im_end|><|im_start|>assistant<|im_sep|>'
        self.retrieval_instruction = ' Here are some paragraphs:\n\n'
        self.retrieval_instruction_late = 'Please find information that are relevant to the following query in the paragraphs above.\n\nQuery: '

        # layer info
        self.num_layer = self.llm.config.num_hidden_layers
        self.head_set = head_set

    def rerank(self, query, documents):
        # FP with query
        llm_prompt, doc_span, query_start_idx, query_end_idx = self.prepare_input_for_document_retrieval(query, documents)
        tok_scores, kv_cache = self.score_documents(llm_prompt, doc_span, query_start_idx, query_end_idx, return_cache=True)

        for i in range(len(kv_cache.key_cache)):
            kv_cache.key_cache[i] = kv_cache.key_cache[i][:,:,:query_start_idx,:]
            kv_cache.value_cache[i] = kv_cache.value_cache[i][:,:,:query_start_idx,:]
        kv_cache._seen_tokens = query_start_idx

        # FP with content-free query
        llm_prompt, doc_span, query_start_idx, query_end_idx = self.prepare_input_for_document_retrieval('N/A', documents)
        tok_scores_na = self.score_documents(llm_prompt, doc_span, query_start_idx, query_end_idx, kv_cache=kv_cache, context_start_idx=query_start_idx)

        del kv_cache
        torch.cuda.empty_cache()

        doc_scores = torch.zeros(len(documents))
        _i = 0
        for tok_score, tok_score_na in zip(tok_scores, tok_scores_na):
            calibrated_score = tok_score - tok_score_na
            threshold = calibrated_score.mean() - 2*calibrated_score.std()
            tok_mask = (calibrated_score>threshold)

            tok_score = tok_score * tok_mask
            tok_score_na = tok_score_na * tok_mask
            doc_scores[_i] = (tok_score - tok_score_na).sum().to('cpu')
            _i += 1

        del tok_score, tok_score_na, tok_scores, tok_scores_na, calibrated_score, threshold, tok_mask
        torch.cuda.empty_cache()

        sorted_results = torch.sort(doc_scores, descending=True)
        return sorted_results.indices.tolist(), sorted_results.values.tolist()

    def score_documents(
            self,
            llm_input,
            doc_span,
            query_start_tok_idx,
            query_end_tok_idx,
            context_start_idx=0,
            return_cache=False,
            kv_cache=None,
        ):

        tokenized_input = self.tokenizer(llm_input,return_tensors='pt').to(self.llm.device)
        _input_ids = tokenized_input.input_ids[:, context_start_idx:]
        _query_indices = list(range(query_start_tok_idx-context_start_idx, query_end_tok_idx-context_start_idx+1))

        if kv_cache is None:
            kv_cache=DynamicCacheWithQuery(query_indices=_query_indices)
        else:
            kv_cache.query_cache = []
            kv_cache._query_indices = _query_indices

        with torch.no_grad():
            output = self.llm(
                input_ids=_input_ids,
                use_cache=True,
                past_key_values=kv_cache,
                output_attentions=True
                )
        kv_cache = output.past_key_values

        del tokenized_input, _input_ids, output
        torch.cuda.empty_cache()

        # loop through all layers and compute attention scores
        all_key_cache = []
        all_query_cache = []
        for i in range(self.num_layer):
            all_key_cache.append(kv_cache.key_cache[i][:,:,:query_end_tok_idx+1].squeeze(0))
            all_query_cache.append(kv_cache.query_cache[i].squeeze(0))

        if not return_cache:
            del kv_cache
            torch.cuda.empty_cache()

        # attention score from all heads
        if self.head_set is None:
            attn_weights = []
            for i in range(self.num_layer):
                attn_weights.append((self.get_attn_all(all_key_cache[i], all_query_cache[i])).mean(-2))
            del all_key_cache, all_query_cache
            torch.cuda.empty_cache()
            attn_weights = torch.stack(attn_weights)
            attn_weights = attn_weights.sum(0).sum(0)

        # attention score from retrieval heads
        else:
            all_key_cache = torch.stack(all_key_cache).squeeze(1)
            all_query_cache = torch.stack(all_query_cache).squeeze(1)
            attn_weights = self.get_attn_head(all_key_cache, all_query_cache)
            attn_weights = attn_weights.mean(-2).sum(0)

        per_doc_results = [None for _ in range(len(doc_span))]
        for i, doc_span in enumerate(doc_span):
            per_doc_results[i] = attn_weights[doc_span[0]:doc_span[1]+1].to('cpu')

        del attn_weights
        torch.cuda.empty_cache()

        if return_cache:
            return per_doc_results, kv_cache
        else:
            return per_doc_results

    def get_attn_all(self, key_states, query_states):
        num_heads, q_len, head_dim = query_states.size()
        num_key_value_heads = key_states.size(0)
        num_key_value_groups = num_heads // num_key_value_heads
        kv_seq_len = key_states.size(-2)

        # expand key head to match query head
        key_states = key_states.unsqueeze(1).expand(num_key_value_heads, num_key_value_groups, kv_seq_len, head_dim)
        key_states = key_states.reshape(num_heads, kv_seq_len, head_dim)

        attn_weights = torch.matmul(query_states, key_states.transpose(-2,-1)) / math.sqrt(head_dim)

        del key_states, query_states
        torch.cuda.empty_cache()

        # apply mask
        causal_mask = torch.ones_like(attn_weights.transpose(-1,-2))
        causal_mask = torch.triu(causal_mask, diagonal=-(kv_seq_len-q_len))
        causal_mask = causal_mask.transpose(-1,-2)
        causal_mask = (1-causal_mask) * torch.finfo(causal_mask.dtype).min
        attn_weights += causal_mask
        attn_lses = torch.logsumexp(attn_weights, dim=-1, keepdim=True)
        attn_weights = torch.exp(attn_weights - attn_lses)

        del causal_mask, attn_lses
        torch.cuda.empty_cache()

        return attn_weights

    def get_attn_head(self, key_states, query_states):
        num_layers, num_heads, q_len, head_dim = query_states.size()
        num_key_value_heads = key_states.size(1)
        num_key_value_groups = num_heads // num_key_value_heads
        kv_seq_len = key_states.size(-2)

        # expand key head to match query head
        key_states = key_states.unsqueeze(2).expand(num_layers, num_key_value_heads, num_key_value_groups, kv_seq_len, head_dim)
        key_states = key_states.reshape(num_layers, num_heads, kv_seq_len, head_dim)

        # extract retrieval heads
        indices = torch.tensor(self.head_set).to('cuda')
        key_states = key_states[indices[:,0], indices[:,1]]
        query_states = query_states[indices[:,0], indices[:,1]]
        del indices
        torch.cuda.empty_cache()

        # compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(-2,-1)) / math.sqrt(head_dim)
        del key_states, query_states
        torch.cuda.empty_cache()

        # apply mask
        causal_mask = torch.ones_like(attn_weights.transpose(-1,-2))
        causal_mask = torch.triu(causal_mask, diagonal=-(kv_seq_len-q_len))
        causal_mask = causal_mask.transpose(-1,-2)
        causal_mask = (1-causal_mask) * torch.finfo(causal_mask.dtype).min
        attn_weights += causal_mask
        attn_lses = torch.logsumexp(attn_weights, dim=-1, keepdim=True)
        attn_weights = torch.exp(attn_weights - attn_lses)

        del causal_mask, attn_lses
        torch.cuda.empty_cache()

        return attn_weights

    def prepare_input_for_document_retrieval(self, query, documents):
        doc_span = []
        query_start_idx = None
        query_end_idx = None

        llm_prompt = self.prompt_prefix + self.retrieval_instruction

        for i, doc in enumerate(documents):

            llm_prompt += f'[document {i+1}]'
            start_len = len(self.tokenizer(llm_prompt).input_ids)

            llm_prompt += ' ' + doc
            end_len = len(self.tokenizer(llm_prompt).input_ids) - self.off_set

            doc_span.append((start_len, end_len))
            llm_prompt += '\n\n'

        start_len = len(self.tokenizer(llm_prompt).input_ids)

        llm_prompt += self.retrieval_instruction_late + f'{query.strip()}'
        end_len = len(self.tokenizer(llm_prompt).input_ids) - self.off_set
        llm_prompt += self.prompt_suffix

        query_start_idx = start_len
        query_end_idx = end_len

        return llm_prompt, doc_span, query_start_idx, query_end_idx
