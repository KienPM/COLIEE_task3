""" Create by Ken at 2020 Dec 23 """
import os
import argparse
import glob
import xml.etree.ElementTree as ET
import re
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
    default=27017,
    help='Mongo DB port'
)
arg_parser.add_argument(
    '--db_name',
    type=str,
    default='COLIEE_Task3',
    help='DB name'
)
arg_parser.add_argument(
    '--collection',
    type=str,
    default='training_data',
    help='Collection name'
)
arg_parser.add_argument(
    '--ignore_auth',
    type=bool,
    action="store_false",
    help='Do authenticate or not'
)
arg_parser.add_argument(
    '--train_data_dir',
    type=str,
    default='/media/ken/Temp/TrainingData/COLIEE_Task3/COLIEE2020statute_data-English/train',
    help='Path to training data directory'
)

MONGO_USER = os.getenv('MONGO_USER', 'COLIEE_Task3')
MONGO_PASS = os.getenv('MONGO_PASS', 'abc13579')
article_re = re.compile(r'\nArticle\s*([0-9]+(-\d+)?)\n')


def process_file(file):
    tree = ET.parse(file)
    root = tree.getroot()
    for pair in root.findall('pair'):
        id_ = pair.get('id')
        articles = pair.find('t1').text
        articles = [match[0] for match in article_re.findall(articles)]
        query = pair.find('t2').text.strip()
        collection.insert_one({
            "pair_id": id_,
            "query": query,
            "relevant": articles
        })


if __name__ == '__main__':
    args = arg_parser.parse_args()
    db_host = args.db_host
    db_port = args.db_port
    db_name = args.db_name
    db_collection = args.collection
    train_data_dir = args.train_data_dir

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

    files = glob.glob(train_data_dir + '/*.xml')
    for f in files:
        try:
            # print(f)
            process_file(f)
        except Exception as e:
            print(f)
            raise e
