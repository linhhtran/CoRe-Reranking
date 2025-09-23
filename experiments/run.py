import os

def run(llm, data, top_k):
    os.system(f'bash srun.sh {llm} {data} {top_k}')

llm_list = ['llama', 'phi', 'mistral', 'granite']

data_list = [
    'trec-covid', 'nfcorpus', 'dbpedia-entity', 'scifact', 'scidocs', 'fiqa', 'nq',
    'fever', 'climate-fever', 'hotpotqa', 'webis-touche2020', 'msmarco', 'quora', 'arguana',
    'cqadupstack-android', 'cqadupstack-english', 'cqadupstack-gaming', 'cqadupstack-gis',
    'cqadupstack-mathematica', 'cqadupstack-physics', 'cqadupstack-programmers', 'cqadupstack-stats',
    'cqadupstack-tex', 'cqadupstack-unix', 'cqadupstack-webmasters', 'cqadupstack-wordpress',
    'mldr_de', 'mldr_en', 'mldr_es', 'mldr_fr', 'mldr_it', 'mldr_pt'
]

for llm in llm_list:
    for data in data_list:
        run(llm, data, 40)
