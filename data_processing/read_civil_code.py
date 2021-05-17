""" Create by Ken at 2020 Dec 23 """
import os
import argparse
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
    default='COLIEE_Task3_BERT',
    help='DB name'
)
arg_parser.add_argument(
    '--collection',
    type=str,
    default='civil_code',
    help='Collection name'
)
arg_parser.add_argument(
    '--ignore_auth',
    action="store_true",
    help='Ignore authenticate or not'
)
arg_parser.add_argument(
    '--data_file',
    type=str,
    help='Path to Civil Code file'
)

MONGO_USER = os.getenv('MONGO_USER', 'COLIEE_Task3')
MONGO_PASS = os.getenv('MONGO_PASS', 'abc13579')

part_re = re.compile(r'^Part\s*([MDCLXVI]+)\s')
chapter_re = re.compile(r'^Chapter\s*([MDCLXVI]+)\s')
section_re = re.compile(r'^Section\s*([0-9]+(-\d+)?)\s')
article_re = re.compile(r'^Article\s*([0-9]+(-\d+)?)\s')
deleted_articles_re = re.compile(r'^Articles.*(to|through|and).*Deleted$')


def run():
    file = open(data_file)
    cur_article = ""
    cur_title = ""
    prev_title = ""
    content = []
    lines = file.readlines()
    # i = 0
    # n = len(lines)
    # while i < n:
    for i, line in enumerate(lines):
        if i % 100 == 0:
            print(i)

        line = line.strip()
        if part_re.search(line) or chapter_re.search(line) \
                or section_re.search(line) or deleted_articles_re.match(line):
            continue

        if line[0] == '(' and line[-1] == ')':
            cur_title = line[1:-1]
            continue

        check_article = article_re.search(line)
        if check_article:
            if cur_article != "":
                collection.insert({
                    "code": cur_article,
                    "title": prev_title,
                    "content": '\n'.join(content)
                })

            content = []
            cur_article = check_article.groups()[0]
            prev_title = cur_title
            cur_title = ''
            line = line[check_article.span()[1] + 1:]

        content.append(line)


if __name__ == '__main__':
    args = arg_parser.parse_args()
    db_host = args.db_host
    db_port = args.db_port
    db_name = args.db_name
    db_collection = args.collection
    data_file = args.data_file

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

    run()
