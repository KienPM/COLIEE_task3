"""
Create by Ken at 2021 Jan 02
Convert text to int sequence
"""
import os
import sys
import argparse
from pymongo import MongoClient
from tqdm import tqdm
from nltk import sent_tokenize

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
    default='COLIEE_Task3',
    help='MongoDB DB name'
)
arg_parser.add_argument(
    '--db_input_collection',
    type=str,
    default='civil_code',
    help='MongoDB input collection name'
)
arg_parser.add_argument(
    '--ignore_auth',
    type=bool,
    action="store_false",
    help='Do authenticate or not'
)
arg_parser.add_argument(
    '--dict_file',
    type=str,
    default='output/dict.tsv',
    help='Path to dict file'
)

MONGO_USER = os.getenv('MONGO_USER', 'COLIEE_Task3')
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


def process_article(content):
    """
    Convert text to int sequence
    :type content: str
    :rtype: list[list[int]]
    """
    lines = content.split('\n')
    res = []
    for line in lines:
        sentences = sent_tokenize(line)
        for s in sentences:
            seq = sentence_to_seq(s, text_to_seq_dict)
            if seq is not None:
                res.append(seq)

    return res


def process_doc(document):
    seq_title = sentence_to_seq(document["title"], text_to_seq_dict)
    seq_content = process_article(document["content"])

    input_collection.update_one(
        {"_id": document["_id"]},
        {
            "$set": {
                "seq_title": seq_title,
                "seq_content": seq_content
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
    input_collection = db[args.db_input_collection]

    text_to_seq_dict = parse_dict_file()

    docs = list(input_collection.find())
    for doc in tqdm(docs):
        process_doc(doc)
