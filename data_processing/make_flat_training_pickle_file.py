"""
Create by Ken at 2021 Jan 05
Create pickle file for training
Data structure for each example: [query sequence, articles, labels corresponding to articles (1~positive, 0~negative)]
"""
import os
import argparse
import sys
import numpy as np
import random
import pickle
from tqdm import tqdm
from pymongo import MongoClient

sys.path.append(os.path.dirname(os.getcwd()))
from utils.string_util import sentence_to_seq
from utils.padding_utils import pad_query, pad_sentence, pad_article_flat

MONGO_USER = os.getenv('MONGO_USER', 'COLIEE_Task3')
MONGO_PASS = os.getenv('MONGO_PASS', 'abc13579')


def load_all_articles():
    all_articles = []
    dict_ = {}
    docs = civil_code_collection.find()
    for doc in docs:
        doc_code = doc['code'].lower()
        if doc['seq_title'] is not None and len(doc['seq_title']) > 0:
            vec = [doc['seq_title']] + doc['seq_content']
        else:
            vec = doc['seq_content']
        dict_[doc_code] = vec
        all_articles.append(doc_code)

    return all_articles, dict_


def parse_dict_file():
    lines = open(args.dict_file, 'r').readlines()
    dict_ = {}
    for line in lines:
        tokens = line.split('\t')
        id_ = int(tokens[0].strip())
        term = tokens[1].strip()
        dict_[term] = id_
    return dict_


def main():
    text_to_seq_dict = parse_dict_file()
    all_articles, all_articles_map = load_all_articles()
    examples = []
    records = list(negative_sampling_collection.find())
    for record in tqdm(records):
        query = sentence_to_seq(record['query'], text_to_seq_dict)
        query = pad_query(query, max_query_len)
        record_negative = record["negative"][:num_es_negative + 5]
        taken = set(record["positive"] + record_negative)
        num_taken = len(taken)

        for positive in record["positive"]:
            if random_es_negative and len(record_negative) > num_es_negative:
                es_negative = random.sample(record_negative, num_es_negative)
            else:
                es_negative = record_negative[:num_es_negative]

            temp_num_random_negative = num_random_negative + (num_es_negative - len(es_negative))
            random_negative = random.sample(all_articles, temp_num_random_negative + num_taken * 2)
            random_negative = list(set(random_negative) - taken)[:temp_num_random_negative]

            negative = es_negative + random_negative
            len_negative = len(negative)
            assert len_negative == num_es_negative + num_random_negative, 'Not enough random negative'

            articles = [positive] + negative
            articles_seq = []
            for a in articles:
                content = all_articles_map[a]
                a_seq = []
                for s in content:
                    a_seq.extend(pad_sentence(s, max_article_len))
                a_seq = pad_article_flat(a_seq, max_article_len=max_article_len)
                articles_seq.append(a_seq)

            labels = [0] * len(articles)
            labels[0] = 1

            ids = np.arange(len(articles))
            np.random.shuffle(ids)
            examples.append([
                np.asarray(query),
                np.asarray([articles_seq[i] for i in ids]),
                np.asarray([labels[i] for i in ids])
            ])

    os.makedirs("output", exist_ok=True)
    with open("output/flat_training_data.pkl", 'wb') as f:
        pickle.dump(examples, f)
        f.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Make pickle file for training')
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
        help='MongoDB input DB name'
    )
    arg_parser.add_argument(
        '--db_civil_code_collection',
        type=str,
        default='civil_code',
        help='MongoDB civil code collection name'
    )
    arg_parser.add_argument(
        '--db_negative_sampling_collection',
        type=str,
        default='negative_sampling',
        help='MongoDB negative sampling collection name'
    )
    arg_parser.add_argument(
        '--ignore_auth',
        action="store_true",
        help='Ignore authentication'
    )
    arg_parser.add_argument(
        '--num_es_negative',
        type=int,
        default=50,
        help='Number of ES negative articles per example'
    )
    arg_parser.add_argument(
        '--num_random_negative',
        type=int,
        default=50,
        help='Number of negative articles per example'
    )
    arg_parser.add_argument(
        '--max_query_len',
        type=int,
        default=150,
        help='Max query length'
    )
    arg_parser.add_argument(
        '--max_article_len',
        type=int,
        default=1000,
        help='Max article length'
    )
    arg_parser.add_argument(
        '--not_random_es_negative',
        action="store_true",
        help='Random sample ES negative'
    )
    arg_parser.add_argument(
        '--dict_file',
        type=str,
        default='output/dict.tsv',
        help='Path to dict file'
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
    civil_code_collection = db[args.db_civil_code_collection]
    negative_sampling_collection = db[args.db_negative_sampling_collection]
    num_es_negative = args.num_es_negative
    num_random_negative = args.num_random_negative
    max_query_len = args.max_query_len
    max_article_len = args.max_article_len
    random_es_negative = not args.not_random_es_negative
    print('> max_query_len: ', max_query_len)
    print('> max_article_len: ', max_article_len)
    print('> random_es_negative: ', random_es_negative)
    print('> num_es_negative: ', num_es_negative)
    print('> num_random_negative: ', num_random_negative)
    print('> training data collection: ', args.db_negative_sampling_collection)

    main()
