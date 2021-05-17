""" Create by Ken at 2021 Jan 10 """
import os
import argparse
import xml.etree.ElementTree as ET
from pymongo import MongoClient

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
    default='COLIEE_Task3_BERT',
    help='DB name'
)
arg_parser.add_argument(
    '--collection',
    type=str,
    default='test_data',
    help='Collection name'
)
arg_parser.add_argument(
    '--ignore_auth',
    action="store_true",
    help='Ignore authenticate or not'
)
arg_parser.add_argument(
    '--test_data_dir',
    type=str,
    default='../data/COLIEE2020statute_data-English/test',
    help='Path to test data directory'
)

MONGO_USER = os.getenv('MONGO_USER', 'COLIEE_Task3')
MONGO_PASS = os.getenv('MONGO_PASS', 'abc13579')


def read_test_data(file):
    tree = ET.parse(file)
    root = tree.getroot()
    for pair in root.findall('pair'):
        id_ = pair.get('id')
        query = pair.find('t2').text.strip()
        collection.insert_one({
            "pair_id": id_,
            "query": query
        })


def read_ground_truth(file):
    f = open(file)
    lines = f.readlines()
    f.close()

    gt = {}
    for line in lines:
        line = line.strip()
        if line == '':
            return

        parts = line.split()
        pair_id = parts[0]
        article_id = parts[2]

        if pair_id in gt:
            gt[pair_id] += [article_id]
        else:
            gt[pair_id] = [article_id]

    for pair_id in gt:
        collection.update_one(
            {"pair_id": pair_id},
            {
                "$set": {
                    "relevant": gt[pair_id]
                }
            }
        )


if __name__ == '__main__':
    args = arg_parser.parse_args()
    db_host = args.db_host
    db_port = args.db_port
    db_name = args.db_name
    db_collection = args.collection
    test_data_dir = args.test_data_dir

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
    db = mongo_client[db_name]
    collection = db[db_collection]
    collection.delete_many({})

    read_test_data(os.path.join(test_data_dir, 'TestData_en.xml'))
    read_ground_truth(os.path.join(test_data_dir, 'task3_test_labels.txt'))
