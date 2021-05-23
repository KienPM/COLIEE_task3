import os
from pymongo import MongoClient
from tqdm import tqdm
import math

# DB_NAME = 'COLIEE_Task3'
DB_NAME = 'COLIEE_Task3_BERT'
client = MongoClient(
    '203.162.10.123', 5007,
    username='COLIEE_Task3',
    password='abc13579',
    authSource=DB_NAME,
    authMechanism='SCRAM-SHA-1'
)
db = client[DB_NAME]


def statistic_query_length():
    min_query_len = 1e9
    max_query_len = 0
    len_list = [128] + [i for i in range(140, 161, 5)]
    len_list_count = [0] * len(len_list)

    query_list = list(db['test_data'].find({}, {'seq_query': 1})) + list(db['training_data'].find({}, {'seq_query': 1}))
    for record in query_list:
        query_len = len(record['seq_query'])

        min_query_len = min(min_query_len, query_len)
        max_query_len = max(max_query_len, query_len)

        for i in range(len(len_list)):
            if query_len <= len_list[i]:
                len_list_count[i] += 1

    print(f'Min: {min_query_len}')
    print(f'Max: {max_query_len}')

    n = len(query_list)
    for i in range(len(len_list)):
        print(f'Length {len_list[i]} covers {len_list_count[i] / n}')


def statistic_article_flat_len():
    min_article_len = 1e9
    max_article_len = 0

    len_list = [i for i in range(300, 750, 50)]
    len_list_count = [0] * len(len_list)

    records = list(db['civil_code'].find({}, {'seq_title': 1, 'seq_content': 1}))
    for record in tqdm(records):
        if record['seq_title']:
            article_len = len(record['seq_title'])
        else:
            article_len = 0
        for vec in record['seq_content']:
            article_len += len(vec)

        min_article_len = min(min_article_len, article_len)
        max_article_len = max(max_article_len, article_len)

        for i in range(len(len_list)):
            if article_len <= len_list[i]:
                len_list_count[i] += 1

    print(f'Min: {min_article_len}')
    print(f'Max: {max_article_len}')

    n = len(records)
    for i in range(len(len_list)):
        print(f'Length {len_list[i]} covers {len_list_count[i] / n}')


if __name__ == '__main__':
    statistic_article_flat_len()
