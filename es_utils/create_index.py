# -*- coding: utf-8 -*-
""" Create by Ken at 2020 Feb 06 """
import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
import argparse

from elasticsearch import Elasticsearch
from elasticsearch_dsl import document, Text, Keyword
from elasticsearch_dsl.connections import connections
from tqdm import tqdm
from pymongo import MongoClient
from utils.string_util import remove_numbering, pre_process_text

MONGO_USER = os.getenv('MONGO_USER', 'COLIEE_Task3')
MONGO_PASS = os.getenv('MONGO_PASS', 'abc13579')

arg_parser = argparse.ArgumentParser(description='Create index')

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
    '--db_collection',
    type=str,
    default='civil_code',
    help='MongoDB civil code collection name'
)

arg_parser.add_argument(
    '--do_auth',
    type=bool,
    default=True,
    help='Do authenticate or not'
)

arg_parser.add_argument(
    '-s',
    '--similarity',
    type=str,
    default="BM25",
    help='Similarity for scoring: {"BM25", "tf_idf", "boolean"}'
)

args = arg_parser.parse_args()
do_auth = args.do_auth
if do_auth:
    mongo_client = MongoClient(
        args.db_host, args.db_port,
        username=MONGO_USER,
        password=MONGO_PASS,
        authSource=args.db_name,
        authMechanism='SCRAM-SHA-1'
    )
else:
    mongo_client = MongoClient(
        args.db_host, args.db_port,
    )
db = mongo_client[args.db_name]
collection = db[args.db_collection]
similarity = args.similarity
index_name = 'coliee_bm25_index'
print("Index name: ", index_name)


class CivilArticle(document.DocType):
    code = Keyword(similarity="boolean")
    title = Text(similarity=similarity)
    raw_title = Text()
    content = Text(similarity=similarity)
    raw_content = Text()

    class Meta:
        """
        Phải khai báo lại class Meta với doc_type và tên index đúng như dưới
        """
        doc_type = 'CivilArticle'
        index = index_name

    class Index:
        """
        Từ bản 0.10.2 trở đi phải khai báo thêm cả class Index có thuộc tính name là tên index trong elastic search như dưới đây
        """
        name = index_name


# Tạo kết nối
es = Elasticsearch()
connections.create_connection(hosts=['localhost'], timeout=20)
connections.add_connection('CivilArticle', es)  # Thêm một doc_type có tên vbpl vào connections
CivilArticle.init(index_name)  # Khởi tạo class Doctype với tên Index


def preprocess_content(content):
    res = []
    lines = content.split('\n')
    for line in lines:
        line = remove_numbering(line)
        words = pre_process_text(line)
        res.append(' '.join(words))

    return ' '.join(res)


def insert(doc):
    civil_article = CivilArticle(code=doc['code'])
    civil_article.title = ' '.join(pre_process_text(doc['title']))
    civil_article.raw_title = doc['title']
    civil_article.content = preprocess_content(doc['content'])
    civil_article.raw_content = doc['content']
    civil_article.save()


if __name__ == '__main__':
    docs = list(collection.find())
    for d in tqdm(docs):
        insert(d)
