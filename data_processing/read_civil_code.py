""" Create by Ken at 2020 Dec 23 """
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
    default='COLIEE_Task3',
    help='DB name'
)
arg_parser.add_argument(
    '--collection',
    type=str,
    default='civil_code',
    help='Collection name'
)
arg_parser.add_argument(
    '--data_file',
    type=str,
    default='/media/ken/Temp/TrainingData/COLIEE_Task3/COLIEE2020statute_data-English/text/civil_code_en-1to724-2.txt',
    help='Path to Civil Code file'
)

article_re = re.compile(r'^Article\s*([0-9]+(-\d+)?)\s')
chapter_re = re.compile(r'^Chapter\s*([MDCLXVI]+)\s')


def run():
    file = open(data_file)
    count = 0
    cur_article = ""
    content = []
    for line in file:
        count += 1
        if count % 100 == 0:
            print(count)

        if chapter_re.search(line):
            continue

        check_article = article_re.search(line)
        if check_article:
            if cur_article != "":
                collection.insert({
                    "code": cur_article,
                    "content": ''.join(content)
                })

            content = []
            cur_article = check_article.groups()[0]
            line = line[check_article.span()[1] + 1:]

        content.append(line)


if __name__ == '__main__':
    args = arg_parser.parse_args()
    db_host = args.db_host
    db_port = args.db_port
    db_name = args.db_name
    db_collection = args.collection
    data_file = args.data_file

    mongo_client = MongoClient(db_host, db_port)
    db = mongo_client[db_name]
    collection = db[db_collection]
    collection.delete_many({})

    run()
