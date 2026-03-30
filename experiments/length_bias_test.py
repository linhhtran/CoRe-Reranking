"""
Length Bias Test for Document Reranking
========================================
Tests whether long irrelevant documents score higher than short relevant ones,
and whether contextual calibration helps mitigate this effect.

Sample construction
-------------------
For each query we compute:

    ratio = max_relevant_length / max_irrelevant_length   (word counts after truncation)

A smaller ratio means the gold document is shorter relative to the longest
irrelevant document, i.e. a stronger length-bias condition.  Queries are kept
when ratio <= --ratio_thresh.

Rerankers evaluated
-------------------
ICR  : all attention heads (no head selection)
QR   : heads selected by the QR (query-relevance) detector
CoRe : heads selected by the CoRe (contrastive retrieval) detector

For each reranker we evaluate both:
  uncalib : raw attention score (no contextual calibration)
  calib   : score after contextual calibration (subtract N/A-query scores)

The two forward passes needed for calibration are shared across all three
rerankers, so the total cost is 2 × forward passes per query sample.

Metrics
-------
NDCG@10      : standard ranking quality
Gold-above-longest (GAL) :
    Fraction of queries where the best-ranked relevant document appears
    before the longest irrelevant document in the reranked list.
    A length-biased reranker will score low here; calibration should help.
"""

import json
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from pyserini.search import get_qrels
from beir.retrieval.evaluation import EvaluateRetrieval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_NAME = {
    'granite': 'ibm-granite/granite-3.2-8b-instruct',
    'llama':   'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'phi':     'microsoft/phi-4',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
}

parser = argparse.ArgumentParser(description='Length bias test for document reranking.')
parser.add_argument('--llm',           type=str,   default='llama',
                    help='Only llama and mistral have hardcoded QR heads')
parser.add_argument('--data',          type=str,   default='fever',
                    help='BEIR dataset name (must have a file in --retriever_dir)')
parser.add_argument('--retriever_dir', type=str,   default='../retriever_output/granite-embedding',
                    help='Directory containing retriever output JSON files')
parser.add_argument('--num_head',      type=int,   default=8,
                    help='Number of top retrieval heads to use for QR and CoRe')
parser.add_argument('--prune',         type=float, default=0.0,
                    help='Layer pruning ratio used when loading CoRe heads')
parser.add_argument('--ratio_thresh',  type=float, default=0.5,
                    help='Keep queries where max_rel_len / max_irrel_len <= this value')
parser.add_argument('--min_samples',   type=int,   default=1,
                    help='Minimum number of biased samples required to proceed')
parser.add_argument('--output_dir',    type=str,   default='../length_bias_output')
args = parser.parse_args()

# Temperature follows the same convention as the main reranking script
args.temp = 0.1 if ('phi' in args.llm or 'llama' in args.llm) else 0.001

TRUNCATE_WORDS = 500

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_head_set(head_file, num_head):
    """Load a head JSON file and return the top-num_head [layer, head] pairs."""
    head_list = json.load(open(head_file))
    scored = [
        ([int(x) for x in key.split('-')], np.mean(val))
        for key, val in head_list.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [pair[0] for pair in scored[:num_head]]


def truncate(text, max_words=TRUNCATE_WORDS):
    return ' '.join(text.split()[:max_words])


def word_count(text):
    return len(text.split())


def get_qrels_for_dataset(data):
    if data == 'msmarco':
        return get_qrels('msmarco-passage-dev-subset')
    return get_qrels(f'beir-v1.0.0-{data}-test')


def find_length_biased_samples(data, retriever_dir, ratio_thresh):
    """
    Load retriever output and qrels, then return:
      - samples          : list of query dicts where ratio <= ratio_thresh,
                           sorted by ratio (most biased first)
      - qrels            : dict {qid: {doc_id: relevance}} (positive only)
      - longest_irrel_id : dict {qid: doc_id} mapping each query to the id of
                           the longest irrelevant document in the retrieved set

    ratio = max_relevant_length / max_irrelevant_length  (after truncation).
    """
    retriever_file = os.path.join(retriever_dir, f'{data}.json')
    if not os.path.exists(retriever_file):
        raise FileNotFoundError(f'Retriever output not found: {retriever_file}')

    query_set = json.load(open(retriever_file))
    qrels_raw = get_qrels_for_dataset(data)

    candidates = []   # (ratio, query, qrel_entry, longest_irrel_doc_id)

    for query in query_set:
        qid = str(query['idx'])

        try:
            rel_raw = qrels_raw[qid]
        except KeyError:
            try:
                rel_raw = qrels_raw[int(qid)]
            except (KeyError, ValueError):
                continue

        positive_ids = {str(k) for k, v in rel_raw.items() if int(v) > 0}
        if not positive_ids:
            continue

        paragraphs = query['paragraphs']
        max_rel_wc    = 0
        max_irrel_wc  = 0
        longest_irrel = None   # doc_id of longest irrelevant

        for p in paragraphs:
            text = str(p['paragraph_text']) if not isinstance(p['paragraph_text'], str) \
                   else p['paragraph_text']
            wc = word_count(truncate(text))   # same truncation as reranking.py
            if str(p['idx']) in positive_ids:
                max_rel_wc = max(max_rel_wc, wc)
            else:
                if wc > max_irrel_wc:
                    max_irrel_wc  = wc
                    longest_irrel = str(p['idx'])

        # Need at least one retrieved relevant doc and one irrelevant doc
        if max_rel_wc == 0 or max_irrel_wc == 0:
            continue

        ratio = max_rel_wc / max_irrel_wc
        if ratio <= ratio_thresh:
            qrel_entry = {str(k): int(v) for k, v in rel_raw.items() if int(v) > 0}
            candidates.append((ratio, query, qrel_entry, longest_irrel))

    # Sort so the most length-biased queries come first
    candidates.sort(key=lambda x: x[0])

    samples          = [c[1] for c in candidates]
    qrels_out        = {str(c[1]['idx']): c[2] for c in candidates}
    longest_irrel_id = {str(c[1]['idx']): c[3] for c in candidates}

    return samples, qrels_out, longest_irrel_id


def build_retrieval_result(paragraphs, sorted_ids, sorted_scores):
    """Convert sorted indices into {doc_id: score} dict expected by EvaluateRetrieval."""
    return {
        str(paragraphs[doc_i]['idx']): float(score)
        for doc_i, score in zip(sorted_ids, sorted_scores)
    }


def gold_above_longest(results_for_condition, qrels, longest_irrel_id):
    """
    Compute the fraction of queries where the best-ranked relevant document
    appears before the longest irrelevant document.

    Args:
        results_for_condition : {qid: {doc_id: score}}  (higher score = ranked higher)
        qrels                 : {qid: {doc_id: rel}}
        longest_irrel_id      : {qid: doc_id}

    Returns:
        float in [0, 1]
    """
    n_total = 0
    n_gold_above = 0

    for qid, doc_scores in results_for_condition.items():
        target_irrel = longest_irrel_id.get(qid)
        if target_irrel is None or target_irrel not in doc_scores:
            continue

        rel_ids = set(qrels.get(qid, {}).keys())
        # Rank documents by score descending (rank 0 = highest)
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        rank_map = {doc_id: rank for rank, (doc_id, _) in enumerate(ranked)}

        irrel_rank = rank_map[target_irrel]
        # Best rank among relevant docs (lowest rank index)
        rel_ranks  = [rank_map[d] for d in rel_ids if d in rank_map]
        if not rel_ranks:
            continue

        best_rel_rank = min(rel_ranks)
        n_total += 1
        if best_rel_rank < irrel_rank:   # gold comes before the longest irrelevant
            n_gold_above += 1

    return n_gold_above / n_total if n_total > 0 else float('nan')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('-' * 60)
    print(f'Length bias test | dataset: {args.data} | LLM: {args.llm}')
    print(f'Ratio threshold  : max_rel_len / max_irrel_len <= {args.ratio_thresh}')

    # ---- Find length-biased samples ------------------------------------
    print('\nSearching for length-biased samples...')
    samples, qrels, longest_irrel_id = find_length_biased_samples(
        args.data, args.retriever_dir, args.ratio_thresh
    )
    print(f'Found {len(samples)} samples with ratio <= {args.ratio_thresh} '
          f'(sorted most-biased first).')

    if len(samples) < args.min_samples:
        print(
            f'Too few samples (need >= {args.min_samples}). '
            'Try raising --ratio_thresh or switching --data.'
        )
        return

    # ---- Load head sets ------------------------------------------------

    core_head_file = f'../head_data/{args.llm}/core_temp{args.temp}_prune{args.prune}.json'
    qr_head_file  = f'../head_data/{args.llm}/qr.json'
    if not os.path.exists(core_head_file):
        raise FileNotFoundError(f'Head file not found: {core_head_file}')
    core_head_set = load_head_set(core_head_file, args.num_head)
    qr_head_set   = load_head_set(qr_head_file, args.num_head)

    print(f'\nTop-{args.num_head} QR   heads: {qr_head_set}')
    print(f'Top-{args.num_head} CoRe heads: {core_head_set}')

    # ---- Initialise multi-reranker ------------------------------------
    from src.multi_reranker import MultiReranker
    reranker = MultiReranker(
        LLM_NAME[args.llm],
        qr_head_set=qr_head_set,
        core_head_set=core_head_set,
        prune=args.prune,
    )

    # ---- Accumulate results -------------------------------------------
    # Structure: results[reranker_name][calib_type][qid] = {doc_id: score}
    reranker_names = ['icr', 'qr', 'core']
    calib_types    = ['uncalib', 'calib']
    results = {
        name: {ct: {} for ct in calib_types}
        for name in reranker_names
    }

    for query in tqdm(samples, desc='Reranking'):
        qid        = str(query['idx'])
        question   = query['question']
        paragraphs = query['paragraphs']

        # Ensure text is a string and apply truncation (mirrors reranking.py)
        for p in paragraphs:
            if not isinstance(p['paragraph_text'], str):
                p['paragraph_text'] = str(p['paragraph_text'])
        for p in paragraphs:
            p['paragraph_text'] = truncate(p['paragraph_text'])

        documents = [p['paragraph_text'].strip() for p in paragraphs]

        # Two forward passes → scores for all 3 rerankers × 2 calib modes
        scores_dict = reranker.rerank_all(question, documents)

        for name in reranker_names:
            for ct in calib_types:
                sorted_ids, sorted_scores = scores_dict[name][ct]
                results[name][ct][qid] = build_retrieval_result(
                    paragraphs, sorted_ids, sorted_scores
                )

    # ---- Evaluate metrics ---------------------------------------------
    evaluator = EvaluateRetrieval()

    header = f'{"Reranker":<10}  {"NDCG@10 (U)":>12}  {"NDCG@10 (C)":>12}  ' \
             f'{"ΔNDCG":>8}  {"GAL (U)":>9}  {"GAL (C)":>9}  {"ΔGAL":>8}'
    sep = '-' * len(header)

    print('\n' + '=' * len(header))
    print(f'Results — {len(samples)} length-biased samples  '
          f'(ratio <= {args.ratio_thresh})')
    print(f'Dataset: {args.data} | LLM: {args.llm}')
    print(f'U = uncalibrated  C = calibrated  '
          f'GAL = gold-above-longest-irrelevant fraction')
    print(sep)
    print(header)
    print(sep)

    summary = {}
    for name in reranker_names:
        row = {}
        for ct in calib_types:
            eval_res    = evaluator.evaluate(qrels, results[name][ct], [10])
            row[f'ndcg_{ct}'] = eval_res[0]['NDCG@10']
            row[f'gal_{ct}']  = gold_above_longest(
                results[name][ct], qrels, longest_irrel_id
            )
        summary[name] = row

        d_ndcg = row['ndcg_calib'] - row['ndcg_uncalib']
        d_gal  = row['gal_calib']  - row['gal_uncalib']
        print(
            f'{name:<10}  '
            f'{row["ndcg_uncalib"]:>12.4f}  {row["ndcg_calib"]:>12.4f}  {d_ndcg:>+8.4f}  '
            f'{row["gal_uncalib"]:>9.4f}  {row["gal_calib"]:>9.4f}  {d_gal:>+8.4f}'
        )
    print('=' * len(header))

    # ---- Save results -------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        f'{args.data}_{args.llm}_ratio{args.ratio_thresh}.json'
    )
    output = {
        'config': vars(args),
        'num_samples': len(samples),
        'metrics': summary,
    }
    json.dump(output, open(output_path, 'w'), indent=2)
    print(f'\nSaved summary to {output_path}')


if __name__ == '__main__':
    main()
