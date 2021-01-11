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
    '--k',
    type=int,
    default=20,
    help='k for recall evaluating'
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
k = args.k


def get_top_k(text_query):
    text_query = ' '.join(pre_process_text(text_query))

    es = Elasticsearch()
    connections.add_connection('CivilArticle', es)
    es_index = 'coliee_bm25_index'

    es_query = {"multi_match": {"query": text_query, "fields": ["title", "content"]}}
    res = es.search(index=es_index, body={"from": 0, "size": k, "query": es_query})
    res = res['hits']['hits']

    articles = [article['_source']["code"].lower() for article in res]

    return articles


examples = list(test_data_collection.find())
os.makedirs('output', exist_ok=True)
output = open(f'output/pre_fetched_bm25_{k}_results.csv', 'w')
for example in tqdm(examples):
    query = example["query"]
    articles = get_top_k(query)
    output.write("{},{}\n".format(example["pair_id"], '|'.join(articles)))

output.close()
