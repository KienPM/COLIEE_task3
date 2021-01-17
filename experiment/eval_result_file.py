""" Create by Ken at 2021 Jan 17 """
import argparse

arg_parser = argparse.ArgumentParser(description='Encode articles')

arg_parser.add_argument(
    '--rel_file',
    type=str,
    default='/home/mlalab/dev/COLIEE_task3/data/COLIEE2020statute_data-English/test/task3_test_labels.txt',
    help='Path to relevant file'
)

arg_parser.add_argument(
    '--result_file',
    type=str,
    help='Path to result file'
)

arg_parser.add_argument(
    '--top_k',
    type=int,
    default=1,
    help='Top k results from result file used for evaluation'
)

args = arg_parser.parse_args()
rel_file = args.rel_file
result_file = args.result_file
top_k = args.top_k


def parse_file(file):
    f = open(file)
    lines = f.readlines()
    f.close()

    data = {}
    for line in lines:
        line = line.strip()
        if line == '':
            return

        parts = line.split()
        pair_id = parts[0]
        article_id = parts[2]

        if pair_id in data:
            data[pair_id] += [article_id]
        else:
            data[pair_id] = [article_id]

    return data


def evaluate():
    rel = parse_file(rel_file)
    result = parse_file(result_file)

    total_precision = 0
    total_recall = 0
    total_f2 = 0
    for pair_id in rel:
        gt = set(rel[pair_id])
        predicted = set(result[pair_id][:top_k])

        true_positive = len(predicted & gt)
        precision = true_positive / len(predicted)
        recall = true_positive / len(gt)
        if precision > 0 or recall > 0:
            f2 = (5 * precision * recall) / (4 * precision + recall)
        else:
            f2 = 0

        total_precision += precision
        total_recall += recall
        total_f2 += f2

    num_examples = len(rel)
    print(f'F2: {total_f2 / num_examples}')
    print(f'Precision: {total_precision / num_examples}')
    print(f'Recall: {total_recall / num_examples}')


if __name__ == '__main__':
    evaluate()
