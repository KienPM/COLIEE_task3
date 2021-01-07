""" Create by Ken at 2020 Dec 28 """
import os
import sys
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from pymongo import MongoClient
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.getcwd()))
from utils.string_util import remove_numbering, pre_process_text

MONGO_USER = os.getenv('MONGO_USER', 'COLIEE_Task3')
MONGO_PASS = os.getenv('MONGO_PASS', 'abc13579')

arg_parser = argparse.ArgumentParser(description='Make stopwords')
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
    '--civil_code_collection',
    type=str,
    default='civil_code',
    help='Civil code collection name'
)
arg_parser.add_argument(
    '--training_data_collection',
    type=str,
    default='training_data',
    help='Training data collection name'
)
arg_parser.add_argument(
    '--ignore_auth',
    action="store_true",
    help='Ignore authenticate or not'
)
arg_parser.add_argument(
    '--min_df',
    type=int,
    default=0,
    help='Min DF'
)
arg_parser.add_argument(
    '--threshold',
    type=float,
    default=1.5,
    help='Threshold to decide if a word is stopwords'
)
arg_parser.add_argument(
    '--output_file',
    type=str,
    default='dict.tsv',
    help='Output file name'
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
civil_code_collection = db[args.civil_code_collection]
training_data_collection = db[args.training_data_collection]
output_file = args.output_file


def load_doc_data():
    documents = []
    records = list(civil_code_collection.find())
    for record in tqdm(records):
        documents.append(record['title'])
        for line in record['content'].split('\n'):
            documents.append(remove_numbering(line))

    return documents


def load_question_data():
    documents = []
    records = list(training_data_collection.find())
    for record in tqdm(records):
        documents.append(record['query'])

    return documents


def process(documents):
    """
    :param documents: list of documents
    """
    vectorizer = CountVectorizer(min_df=args.min_df, tokenizer=pre_process_text)
    vector = vectorizer.fit_transform(documents)
    vector = vector.toarray()
    vector = np.sum(vector, axis=0)

    dictionary = []
    for term in vectorizer.vocabulary_:
        index = vectorizer.vocabulary_[term]
        dictionary.append({
            'id': index + 1,
            'term': term,
            'occurrences': vector[index]
        })

    dictionary = sorted(dictionary, key=lambda x: x['id'])

    os.makedirs('output', exist_ok=True)

    out = open(f'output/{output_file}', 'w')
    for item in dictionary:
        # term, id, number of occurrences
        out.write('{}\t{}\t{}\n'.format(item['id'], item['term'], item['occurrences']))


if __name__ == '__main__':
    docs = load_doc_data() + load_question_data()
    process(docs)
