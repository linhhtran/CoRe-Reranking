#!/bin/bash

python reranking.py --llm $1 --data $2 --top_k $3
