import json
import os
from tqdm import tqdm
import argparse

llm_name = {
    'granite': 'ibm-granite/granite-3.2-8b-instruct',
    'llama': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'phi': 'microsoft/phi-4',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2'
}

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--llm', type=str, default='mistral', choices=['mistral', 'llama', 'phi', 'granite'])
parser.add_argument('--detector', type=str, default='core', choices=['qr', 'core'])
parser.add_argument('--temp', type=float, default=0.001)
parser.add_argument('--prune', type=float, default=0.0)
args = parser.parse_args()

def main():
    print('-'*50)
    print(f'retrieving heads for {args.llm} using {args.detector} detector')

    os.makedirs(f'../head_data/{args.llm}', exist_ok=True)
    if args.detector == 'core':
        output_file = f'../head_data/{args.llm}/{args.detector}_temp{args.temp}_prune{args.prune}.json'
    else:
        output_file = f'../head_data/{args.llm}/{args.detector}.json'
    if os.path.exists(output_file):
        print('run already completed')
        return

    if args.detector == 'qr':
        from src.qr_detector import HeadDetector
        query_set = json.load(open(f'../head_data/nq_qr.json'))
        detector = HeadDetector(llm_name[args.llm])
    elif args.detector == 'core':
        from src.core_detector import HeadDetector
        query_set = json.load(open(f'../head_data/nq_core.json'))
        detector = HeadDetector(llm_name[args.llm], args.temp, args.prune)

    for _, query in enumerate(tqdm(query_set)):
        question = query['question']
        paragraphs = query['paragraphs']

        neg_idx = []
        for i in range(len(paragraphs)):
            if args.detector == 'qr':
                if paragraphs[i]['is_gold']:
                    pos_idx = i
            elif args.detector == 'core':
                if paragraphs[i]['is_positive']:
                    pos_idx = i
                elif paragraphs[i]['is_negative']:
                    neg_idx.append(i)
            if not isinstance(paragraphs[i]['paragraph_text'], str):
                paragraphs[i]['paragraph_text'] = str(paragraphs[i]['paragraph_text'])

        documents = [(p['paragraph_text']).strip() for p in paragraphs]
        detector.compute_retrieval_score(question, documents, pos_idx, neg_idx)

    head_score_list = detector.get_head_score()

    # save detection results
    json.dump(head_score_list, open(output_file, 'w'), indent=2)
    print(f'saved detection results to {output_file}')

if __name__ == '__main__':
    main()
