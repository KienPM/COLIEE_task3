""" Create by Ken at 2021 Jan 10 """
import os
import argparse
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.dirname(os.getcwd()))
from model.han_sparsemax_model import create_model

arg_parser = argparse.ArgumentParser(description='Split model')
arg_parser.add_argument(
    '--model_dir',
    type=str,
    help='Model directory'
)
arg_parser.add_argument(
    '--model_file',
    type=str,
    help='Model file name'
)
args = arg_parser.parse_args()

model_dir = args.model_dir
model_file = args.model_file


def save_models():
    model, query_encoder, article_encoder = create_model()
    model.load_weights(os.path.join(model_dir, model_file))
    query_encoder.save_weights(os.path.join(model_dir, 'query_encoder.h5'))
    article_encoder.save_weights(os.path.join(model_dir, 'article_encoder.h5'))


if __name__ == '__main__':
    save_models()
