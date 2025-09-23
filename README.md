# Attention-based Reranker with Contrastive Retrieval Heads
Repository for attention-based reranker with contrastive retrieval head detection. Our head detector identifies top retrieval heads by contrasting the attention score of positive and hard negative documents. The top detected retrieval heads (roughly 1% of all heads) significantly improves reranking task compared to noisy aggregation over all heads.

The attention-based reranker implementation is adapted from [In-Context-Reranking](https://github.com/OSU-NLP-Group/In-Context-Reranking).

# Datasets
We use [granite-embedding-30m-english](https://huggingface.co/ibm-granite/granite-embedding-30m-english) to retrieve top-40 documents for BEIR benchmark, and [granite-embedding-107m-multilingual](https://huggingface.co/ibm-granite/granite-embedding-107m-multilingual) for MLDR datasets.

We upload the retriever outputs [here](https://drive.google.com/drive/folders/1nYDB1J03g8O9AlU3Zw6d6m2xQ1tTd1aQ?usp=sharing) which can be downloaded and stored in the `./retriever_output` folder.

The head detection data can be downloaded from [here](https://drive.google.com/drive/folders/11CxygqHC_sPoYQHdRSfU-aUrihSuVTki?usp=sharing) which should be stored in the `./head_data` folder.

# Experiment examples
We already include the CoRe head scores for each model in the `./head_data` folder.
The head scores (example for Mistral 7B) can be reproduced with the following command (run from `./experiments/`):

```bash
python head_detection.py --llm mistral --detector core --temp 0.001
```

The following example command runs the reranking process with 8 retrieval heads on hotpotqa dataset:

```bash
python reranking.py --llm mistral --data hotpotqa --reranker core --num_head 8
```