#!/usr/bin/env bash

INDEX_NAME="coliee_bm25_index"

echo "INDEX_NAME: $INDEX_NAME"
echo "SIMILARITY_NAME: BM25"

curl -X PUT "localhost:9200/$INDEX_NAME?pretty" -H 'Content-Type: application/json' -d"
{
  \"settings\": {
    \"number_of_shards\": 1
  }
}"