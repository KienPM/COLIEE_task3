import os
import argparse
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from elasticsearch import Elasticsearch
from elasticsearch_dsl.connections import connections
from pymongo import MongoClient
from utils.string_util import pre_process_text

MONGO_USER = os.getenv('MONGO_USER', 'COLIEE_Task3')
MONGO_PASS = os.getenv('MONGO_PASS', 'abc13579')

arg_parser = argparse.ArgumentParser(description='BM25 experiment')
arg_parser.add_argument(
    '--db_host',
    type=str,
    default='localhost',
    help='MongoDB host'
)
arg_parser.add_argument(
    '--db_port',
    type=int,
    default=5007,
    help='MongoDB port'
)
arg_parser.add_argument(
    '--db_name',
    type=str,
    default='COLIEE_Task3',
    help='MongoDB DB name'
)
arg_parser.add_argument(
    '--test_data_collection',
    type=str,
    default='test_data',
    help='MongoDB test collection name'
)
arg_parser.add_argument(
    '--ignore_auth',
    action='store_true',
    help='Ignore authenticate or not'
)
arg_parser.add_argument(
    '--top_k',
    type=int,
    default=100,
    help='Top k results from result file to save'
)
args = arg_parser.parse_args()

if args.ignore_auth:
    mongo_client = MongoClient(args.db_host, args.db_port)
else:
    mongo_client = MongoClient(
        args.db_host, args.db_port,
        username=MONGO_USER,
        password=MONGO_PASS,
        authSource=args.db_name,
        authMechanism='SCRAM-SHA-1'
    )

db = mongo_client[args.db_name]
test_data_collection = db[args.test_data_collection]
top_k = args.top_k


def get_top_k(text_query):
    text_query = ' '.join(pre_process_text(text_query))

    es = Elasticsearch()
    connections.add_connection('CivilArticle', es)
    es_index = 'coliee_bm25_index'

    es_query = {"multi_match": {"query": text_query, "fields": ["title", "content"]}}
    res = es.search(index=es_index, body={"from": 0, "size": top_k, "query": es_query})
    res = res['hits']['hits']

    articles_scores = [{'article': article['_source']["code"].lower(), 'score': article['_score']} for article in res]

    return articles_scores


if __name__ == '__main__':
    examples = list(test_data_collection.find())
    os.makedirs('output', exist_ok=True)
    output = open(f'output/bm25_{top_k}_results.csv', 'w')
    for example in tqdm(examples):
        query = example["query"]
        articles_scores = get_top_k(query)
        for index, item in enumerate(articles_scores):
            res_line = f"{example['pair_id']} Q0 {item['article']} {index + 1} {item['combined_score']} ken"
            output.write(f'{res_line}\n')

    output.close()
