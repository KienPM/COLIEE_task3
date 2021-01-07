"""
Create by Ken at 2021 Jan 02
Make training data
"""
import os
import argparse
import sys
from tqdm import tqdm
from pymongo import MongoClient

sys.path.append(os.path.dirname(os.getcwd()))

MONGO_USER = os.getenv('MONGO_USER', 'COLIEE_Task3')
MONGO_PASS = os.getenv('MONGO_PASS', 'abc13579')


def parse_experiment_file():
    lines = open(args.experiment_file, 'r').readlines()[:-1]
    dict_ = {}
    for line in lines:
        parts = line.split(',')
        query_id = parts[0]
        exp_out = parts[3].split('|')
        dict_[query_id] = exp_out
    return dict_


def main():
    records = list(training_data_collection.find())
    for record in tqdm(records):
        _id = record["_id"]
        positive = record["relevant"]
        negative = [item for item in experiment_data[str(_id)] if item not in positive]
        output_collection.insert_one({
            "_id": _id,
            "query": record["query"],
            "positive": positive,
            "negative": negative
        })


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Make training data')
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
        '--db_training_data_collection',
        type=str,
        default='training_data',
        help='MongoDB train ground truth collection name'
    )
    arg_parser.add_argument(
        '--experiment_file',
        type=str,
        default='../experiment/output/bm25_300.csv',
        help='Path to experiment file (for getting negative examples)'
    )
    arg_parser.add_argument(
        '--db_output_collection',
        type=str,
        default='negative_sampling',
        help='MongoDB ground truth collection name'
    )
    arg_parser.add_argument(
        '--ignore_auth',
        action="store_true",
        help='Ignore authentication'
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
    training_data_collection = db[args.db_training_data_collection]
    output_collection = db[args.db_output_collection]
    output_collection.delete_many({})

    experiment_data = parse_experiment_file()
    main()
