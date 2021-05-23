import os
import sys
import argparse
from pymongo import MongoClient
from tqdm import tqdm

sys.path.append(os.path.dirname(os.getcwd()))
from utils.string_util import sentence_to_seq

arg_parser = argparse.ArgumentParser(description='Text to sequence')
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
    default='COLIEE_Task3_BERT',
    help='MongoDB DB name'
)
arg_parser.add_argument(
    '--db_test_collection',
    type=str,
    default='test_data',
    help='MongoDB test collection name'
)
arg_parser.add_argument(
    '--db_training_collection',
    type=str,
    default='training_data',
    help='MongoDB training collection name'
)
arg_parser.add_argument(
    '--ignore_auth',
    action="store_true",
    help='Ignore authenticate or not'
)

MONGO_USER = os.getenv('MONGO_USER', 'COLIEE_Task3')
MONGO_PASS = os.getenv('MONGO_PASS', 'abc13579')


def process(collection):
    docs = list(collection.find({}, {'query': 1}))
    for doc in tqdm(docs):
        query = doc['query']
        seq_query = sentence_to_seq(query)
        collection.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {
                    "seq_query": seq_query,
                }
            }
        )


if __name__ == '__main__':
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
    test_collection = db[args.db_test_collection]
    training_collection = db[args.db_training_collection]
    process(test_collection)
    process(training_collection)
