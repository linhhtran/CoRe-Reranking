import os
import json
from datasets import load_dataset
from rank_eval import Qrels, Run, evaluate

llm_list = {
    'granite': 'ibm-granite/granite-3.2-8b-instruct',
    'llama': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2'
}
language_list = ["de", "en", "es", "fr", "it", "pt"]
top_k = 40

all_qrels = {}
for language in language_list:
    hf_dataset = load_dataset('Shitao/MLDR', language, split='test')
    all_qrels[language] = {q['query_id']: q['positive_passages'] for q in hf_dataset}
print('\n\n\n')

def get_score(language, fname):
    if os.path.exists(fname):
        results = json.load(open(fname))
    else:
        print(f'missing {fname}')
        return None

    # load data
    qrels = all_qrels[language]

    # evaluate
    qrels_eval = Qrels()
    run_eval = Run()
    for qid in results:
        # qrel
        doc_ids = []
        scores = []
        for doc in qrels[qid]:
            doc_ids.append(doc['docid'])
            scores.append(1)
        qrels_eval.add(qid, doc_ids, scores)

        # run
        doc_ids = []
        scores = []
        for doc_id in results[qid]:
            doc_ids.append(doc_id)
            scores.append(results[qid][doc_id])
        run_eval.add(qid, doc_ids, scores)

    results = evaluate(qrels_eval, run_eval, ["ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10"])
    return results

for llm in llm_list:
    if 'phi' in llm or 'llama' in llm:
        temp = 0.1
    else:
        temp = 0.001
    for language in language_list:
        fname = f'../reranking_output/{llm}/top{top_k}/mldr_{language}_core_temp{temp}_prune0.0.json'
        ndcg = get_score(language, fname)
        print(f'{llm} {language}:\n\t{ndcg}')