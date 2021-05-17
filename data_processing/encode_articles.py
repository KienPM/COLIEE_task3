""" Create by Ken at 2021 Jan 10"""
import os
import sys
import argparse
from tqdm import tqdm
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from pymongo import MongoClient

sys.path.append(os.path.dirname(os.getcwd()))
from model.han_model import create_model
from utils.padding_utils import pad_sentence, pad_article

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
    '--exp_db_name',
    type=str,
    default='COLIEE_Task3_exp',
    help='Experiment DB name'
)
arg_parser.add_argument(
    '--db_civil_code_collection',
    type=str,
    default='civil_code',
    help='Civil code collection name'
)
arg_parser.add_argument(
    '--ignore_auth',
    action="store_true",
    help='Ignore authenticate or not'
)
arg_parser.add_argument(
    '--max_num_sen',
    type=int,
    default=35,
    help='Max number of sentences per article'
)
arg_parser.add_argument(
    '--max_sen_len',
    type=int,
    default=25,
    help='Max sentence len'
)
arg_parser.add_argument(
    '--exp_name',
    type=str,
    help='Experiment name'
)
args = arg_parser.parse_args()

MONGO_USER = os.getenv('MONGO_USER', 'admin')
MONGO_PASS = os.getenv('MONGO_PASS', 'abc13579')

exp_name = args.exp_name
db_host = args.db_host
db_port = args.db_port
db_name = args.db_name
exp_db_name = args.exp_db_name
db_civil_code_collection = args.db_civil_code_collection
max_num_sen = args.max_num_sen
max_sen_len = args.max_sen_len

_, _, article_encoder = create_model()
article_encoder_weights_path = f'../model/trained_models/{exp_name}/article_encoder.h5'
print(f'>>> Load article encoder weights from {article_encoder_weights_path}')
article_encoder.load_weights(article_encoder_weights_path)
print(f'>>> max_num_sen: {max_num_sen}')
print(f'>>> max_sen_len: {max_sen_len}')

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
db = mongo_client[db_name]
exp_db = mongo_client[args.exp_db_name]
civil_code_collection = db[db_civil_code_collection]
output_collection = exp_db[f'{exp_name}_encoded_articles']
output_collection.delete_many({})

records = list(civil_code_collection.find())
for article in tqdm(records):
    doc_code = article['code']

    if article['seq_title'] is not None and len(article['seq_title']) > 0:
        content = [article['seq_title']] + article['seq_content']
    else:
        content = article['seq_content']
    article_seq = []
    for s in content:
        article_seq.extend(pad_sentence(s, max_sen_len))
    article_seq = pad_article(article_seq, max_num_sen=max_num_sen, max_sen_len=max_sen_len)

    article_seq = np.array(article_seq, dtype='int32')
    article_seq = article_seq[np.newaxis, :]
    article_rep = article_encoder(article_seq)
    article_rep = tf.keras.backend.eval(article_rep)
    article_rep = article_rep[0].tolist()

    output_collection.insert_one({
        'code': doc_code.lower(),
        'vector': article_rep
    })
