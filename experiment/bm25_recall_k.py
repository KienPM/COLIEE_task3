"""
Similarity BM25
Query fields: title, content
"""
import os
import argparse
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from elasticsearch import Elasticsearch
from elasticsearch_dsl.connections import connections
from pymongo import MongoClient
from utils.string_util import pre_process_text
from experiment.ndcg import n_dcg

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
    '--train_data_collection',
    type=str,
    default='training_data',
    help='MongoDB ground truth doc_collection name'
)
arg_parser.add_argument(
    '--do_auth',
    type=bool,
    default=True,
    help='Do authenticate or not'
)
arg_parser.add_argument(
    '--k',
    type=int,
    default=20,
    help='k for recall evaluating'
)
args = arg_parser.parse_args()

if args.do_auth:
    mongo_client = MongoClient(
        args.db_host, args.db_port,
        username=MONGO_USER,
        password=MONGO_PASS,
        authSource=args.db_name,
        authMechanism='SCRAM-SHA-1'
    )
else:
    mongo_client = MongoClient(args.db_host, args.db_port)

db = mongo_client[args.db_name]
ground_truth_collection = db[args.train_data_collection]
k = args.k


def get_top_k_scores(text_query):
    text_query = ' '.join(pre_process_text(text_query))

    es = Elasticsearch()
    connections.add_connection('CivilArticle', es)
    es_index = 'coliee_bm25_index'

    es_query = {"multi_match": {"query": text_query, "fields": ["title", "content"]}}
    res = es.search(index=es_index, body={"from": 0, "size": k, "query": es_query})
    res = res['hits']['hits']
    count = 0

    articles = []
    scores = []
    i = 0
    while i < len(res) and count < k:
        try:
            if res[i]:
                doc_code = res[i]['_source']["code"].lower()
                articles.append(doc_code)
                scores.append(str(res[i]['_score']))
            else:
                break
            i += 1
        except IndexError:
            continue

    return articles, scores


examples = list(ground_truth_collection.find())
os.makedirs('output', exist_ok=True)
output = open(f'output/bm25_{k}.csv', 'w')
total_recall = 0
total_n_dcg = 0
ignore_examples = 0  # Number of examples don't contain article ground truth
for example in tqdm(examples):
    query = example["query"]
    articles, scores = get_top_k_scores(query)

    ground_truth = example["relevant"]

    intersection = set(articles) & set(ground_truth)
    if len(ground_truth) == 0:
        recall = 0
        n_dcg_score = 0
        ignore_examples += 1
    else:
        recall = len(intersection) / len(ground_truth)
        n_dcg_score = n_dcg(articles, ground_truth)

    total_recall += recall
    total_n_dcg += n_dcg_score
    output.write("{},{},{},{},{}\n".format(example["_id"], recall, n_dcg_score, '|'.join(articles), ','.join(scores)))

output.write("Average recall: {},Average NDCG: {}".format(total_recall / (len(examples) - ignore_examples),
                                                          total_n_dcg / (len(examples) - ignore_examples)))
output.close()
