""" Create by Ken at 2021 Jan 16 """
import os
import sys
import argparse
import numpy as np
import time

from tqdm import tqdm
import tensorflow as tf
from pymongo import MongoClient

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.dirname(os.getcwd()))

from model.han_model import create_model
from utils.string_util import sentence_to_seq
from utils.padding_utils import pad_query

tf.get_logger().setLevel('ERROR')

arg_parser = argparse.ArgumentParser(description='Encode articles')
arg_parser.add_argument(
    '--db_host',
    type=str,
    default='localhost',
    help='Mongo DB host'
)
arg_parser.add_argument(
    '--db_port',
    type=int,
    default=5007,
    help='Mongo DB port'
)
arg_parser.add_argument(
    '--db_name',
    type=str,
    default='COLIEE_Task3',
    help='DB name'
)
arg_parser.add_argument(
    '--exp_db_name',
    type=str,
    default='COLIEE_Task3_exp',
    help='Experiment DB name'
)
arg_parser.add_argument(
    '--db_test_data_collection',
    type=str,
    default='test_data',
    help='Test data collection name'
)
arg_parser.add_argument(
    '--ignore_auth',
    action="store_true",
    help='Ignore authenticate or not'
)
arg_parser.add_argument(
    '--es_output_file',
    type=str,
    default='../data/pre_fetched_bm25_500_results.csv',
    help='Path to ES output file'
)
arg_parser.add_argument(
    '--es_output_limit',
    type=int,
    help='Limit number of results from ES output file'
)
arg_parser.add_argument(
    '--es_score_weight',
    type=float,
    default=0.0,
    help='ES score weight when compute final score'
)
arg_parser.add_argument(
    '--exp_name',
    type=str,
    help='Experiment name'
)
arg_parser.add_argument(
    '--dict_file',
    type=str,
    default='../data/dict.tsv',
    help='Path to dict file'
)
arg_parser.add_argument(
    '--top_k',
    type=int,
    default=100,
    help='Top k results from result file to save'
)

MONGO_USER = os.getenv('MONGO_USER', 'admin')
MONGO_PASS = os.getenv('MONGO_PASS', 'abc13579')


def parse_dict_file():
    lines = open(args.dict_file, 'r').readlines()
    dict_ = {}
    for line in lines:
        tokens = line.split('\t')
        id_ = int(tokens[0].strip())
        term = tokens[1].strip()
        dict_[term] = id_
    return dict_


def parse_pre_fetched_bm25_results():
    experiment_data = {}
    lines = open(es_output_file, 'r').readlines()
    for line in lines:
        parts = line.strip().split(',')
        pair_id = parts[0]
        exp_out = parts[1].strip().split('|')[:es_output_limit]
        experiment_data[pair_id] = exp_out
    return experiment_data


def run():
    _, query_encoder, _ = create_model()
    print('Loading query encoder weights from {}...'.format(query_encoder_weights_path))
    query_encoder.load_weights(query_encoder_weights_path)

    if args.ignore_auth:
        mongo_client = MongoClient(args.db_host, args.db_port)
    else:
        mongo_client = MongoClient(
            args.db_host, args.db_port,
            username=MONGO_USER,
            password=MONGO_PASS,
            authSource='admin',
            authMechanism='SCRAM-SHA-1'
        )

    db = mongo_client[args.db_name]
    exp_db = mongo_client[args.exp_db_name]
    test_collection = db[db_test_data_collection]
    encoded_article_collection = exp_db[db_encoded_article_collection]

    encoded_articles = {}
    print('Loading encoded articles from {}...'.format(db_encoded_article_collection))
    for record in tqdm(list(encoded_article_collection.find())):
        doc_code = record['code'].lower()
        encoded_articles[doc_code] = record['vector']

    examples = list(test_collection.find())
    os.makedirs('output', exist_ok=True)
    output_files = open(f'output/{exp_name}_es_w_{es_score_weight}_es_limit_{es_output_limit}_{top_k}_results.txt', 'w')

    start_time = time.time()
    for example in tqdm(examples):
        es_res = es_output[example['pair_id']]
        articles = []
        articles_rep = []
        for article_id in es_res:
            article = encoded_articles[article_id]
            articles.append(article_id)
            articles_rep.append(article)
        articles_rep = np.array(articles_rep)

        query = sentence_to_seq(example["query"], text_to_seq_dict)
        query = pad_query(query, 40)
        query = np.array(query, dtype='int32')
        query = query[np.newaxis, :]
        query_rep = query_encoder(query)

        group_size = len(articles_rep)
        query_rep = tf.tile(query_rep, [group_size, 1])
        scores = tf.keras.layers.dot([query_rep, articles_rep], axes=-1)
        scores = tf.reshape(scores, (group_size,))

        scores = tf.keras.backend.eval(scores)
        articles_scores = []
        num_candidates = len(articles)
        for i in range(num_candidates):
            articles_scores.append({
                'article': articles[i],
                'es_score': num_candidates - i,
                'neural_score': scores[i].item()
            })

        # Sort by neural score
        articles_scores = sorted(articles_scores, key=lambda x: x['neural_score'], reverse=True)

        # Combine ES score and neural score
        for i in range(num_candidates):
            article_id = articles_scores[i]
            score = article_id['es_score'] * es_score_weight + (num_candidates - i) * neural_score_weight
            article_id['combined_score'] = score

        # Sort by combined score
        articles_scores = sorted(articles_scores, key=lambda x: x['combined_score'], reverse=True)
        articles_scores = articles_scores[:top_k]
        for index, item in enumerate(articles_scores):
            res_line = f"{example['pair_id']} Q0 {item['article']} {index + 1} {item['combined_score']} ken"
            output_files.write(f'{res_line}\n')

    output_files.close()
    print(f'Time: {time.time() - start_time}')


if __name__ == '__main__':
    args = arg_parser.parse_args()
    exp_name = args.exp_name
    db_host = args.db_host
    db_port = args.db_port
    db_name = args.db_name
    db_test_data_collection = args.db_test_data_collection
    db_encoded_article_collection = exp_name + '_encoded_articles'
    query_encoder_weights_path = "../model/trained_models/{}/query_encoder.h5".format(exp_name)
    es_output_file = args.es_output_file
    es_output_limit = args.es_output_limit
    es_score_weight = args.es_score_weight
    neural_score_weight = 1 - es_score_weight
    top_k = args.top_k
    print(f'>>> es_output_file: {es_output_file}')
    print(f'>>> es_output_limit: {es_output_limit}')
    print(f'>>> es_score_weight: {es_score_weight}')
    print(f'>>> Dict file: {args.dict_file}')
    print(f'>>> query_encoder_weights_path: {query_encoder_weights_path}')
    print(f'>>> top_k: {top_k}')

    text_to_seq_dict = parse_dict_file()
    es_output = parse_pre_fetched_bm25_results()
    run()
