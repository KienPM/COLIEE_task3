""" Create by Ken at 2020 Dec 28 """
import os
import re
import argparse
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from pymongo import MongoClient
from tqdm import tqdm
import numpy as np

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
    default=27017,
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

mongo_client = MongoClient(args.db_host, args.db_port)
db = mongo_client[args.db_name]
civil_code_collection = db[args.civil_code_collection]
training_data_collection = db[args.training_data_collection]
output_file = args.output_file

clause_re = re.compile(r'^\(\d+\)')


def load_doc_data():
    documents = []
    records = list(civil_code_collection.find())
    for record in tqdm(records):
        documents.append(record['title'] + ' ' + record['content'])

    return documents


def load_question_data():
    documents = []
    records = list(training_data_collection.find())
    for record in tqdm(records):
        documents.append(record['query'])

    return documents


def tokenizer(s):
    return word_tokenize(s)


def pre_process_line(line):
    match_clause = clause_re.match(line)
    if match_clause:
        line = line[match_clause.span()[1] + 1:]
    return line


def process(documents):
    """
    :param documents: list of documents
    """
    vectorizer = CountVectorizer(min_df=args.min_df, tokenizer=tokenizer)
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
    # print(pre_process_line('(2) The exercise of rights and performance of duties must be done in good faith.'))
