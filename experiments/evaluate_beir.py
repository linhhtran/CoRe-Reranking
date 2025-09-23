import os
import json
from pyserini.search import get_qrels
from beir.retrieval.evaluation import EvaluateRetrieval

llm_list = {
    'granite': 'ibm-granite/granite-3.2-8b-instruct',
    'llama': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'phi': 'microsoft/phi-4',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2'
}
beir_list = [
    'trec-covid', 'nfcorpus', 'dbpedia-entity', 'scifact', 'scidocs', 'fiqa', 'nq', 'fever',
    'climate-fever', 'hotpotqa', 'webis-touche2020', 'msmarco', 'quora', 'arguana',
    'cqadupstack-android', 'cqadupstack-english', 'cqadupstack-gaming', 'cqadupstack-gis',
    'cqadupstack-mathematica', 'cqadupstack-physics', 'cqadupstack-programmers', 'cqadupstack-stats',
    'cqadupstack-tex', 'cqadupstack-unix', 'cqadupstack-webmasters', 'cqadupstack-wordpress'
]

top_k = 40

def get_score(data, fname):
    results = json.load(open(fname))
    if data == 'msmarco':
        qrel_name = 'msmarco-passage-dev-subset'
    else:
        qrel_name = f'beir-v1.0.0-{data}-test'
    _qrels = get_qrels(qrel_name)
    evaluator = EvaluateRetrieval()
    qrels = {}

    for qid in results:
        assert isinstance(qid, str)
        try:
            __qrels = _qrels[qid]
        except:
            try:
                __qrels = _qrels[int(qid)]
            except:
                print('Error in qrels for query id: ', qid)
                continue

        # make sure the qrels are in the right format
        qrels[qid] = {}
        for doc_id in __qrels:
            qrels[qid][str(doc_id)] = __qrels[doc_id]

        doc_keys = list(qrels[qid].keys())
        for key in doc_keys:
            if not isinstance(qrels[qid][key], int):
                qrels[qid][key] = int(qrels[qid][key])
            if qrels[qid][key] == 0:
                qrels[qid].pop(key)

    ks = [1,5,10]
    eval_results = evaluator.evaluate(qrels, results, ks)
    return eval_results

if __name__ == '__main__':
    for llm in llm_list.keys():
        print('-'*20, llm, '-'*20)
        if 'phi' in llm or 'llama' in llm:
            temp = 0.1
        else:
            temp = 0.001
        configs = {}
        for prune in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]:
            configs[f'core_temp{temp}_prune{prune}'] = 0
        all_results = {
            'cqadupstack-average': {k:v for k,v in configs.items()},
            'beir-average': {k:v for k,v in configs.items()},
        }

        for data in beir_list:
            print(data)
            for config in configs.keys():
                fname = f'../reranking_output/{llm}/top{top_k}/{data}_{config}.json'
                eval_results = get_score(data, fname)
                all_results['beir-average'][config] += eval_results[0]['NDCG@10']
                if 'cqadupstack' in data:
                    all_results['cqadupstack-average'][config] += eval_results[0]['NDCG@10']
                else:
                    print(f"\t{config}: {eval_results[0]['NDCG@10']:.3f}")
            print('\n')

        for key in all_results['beir-average'].keys():
            all_results['cqadupstack-average'][key] /= 12
            all_results['beir-average'][key] /= 15
            print(f"cqa average ({key}): {all_results['cqadupstack-average'][key]:.3f}")
            print(f"beir average ({key}): {all_results['beir-average'][key]:.3f}")

