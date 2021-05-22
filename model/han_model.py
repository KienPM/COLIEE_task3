"""
Create by Ken at 2021 Jan 11
Split each article into sentences
Encode sentences -> encode article
"""
import os
import sys
import tensorflow as tf
from tensorflow.keras.layers import *
from custom_layers import FeatureExtractor, PaddingMask

sys.path.append(os.path.dirname(os.getcwd()))

MAX_QUERY_LEN = 150
MAX_NUM_SENTENCES = 35
MAX_SENTENCE_LEN = 25
GROUP_SIZE = 31
D_Q = 200

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def log_hyper_parameters(logger):
    logger.info("-" * 10)
    logger.info("Hyper parameters")
    logger.info(f"MAX_QUERY_LEN: {MAX_QUERY_LEN}")
    logger.info(f"MAX_NUM_SENTENCES: {MAX_NUM_SENTENCES}")
    logger.info(f"MAX_SENTENCE_LEN: {MAX_SENTENCE_LEN}")
    logger.info(f"GROUP_SIZE: {GROUP_SIZE}")
    logger.info(f"D_Q: {D_Q}")
    logger.info("-" * 10)


def create_model():
    """
    Return: Model, Query encoder, Article encoder
    """
    feature_extractor = FeatureExtractor()

    # Query branch
    query_input = Input((MAX_QUERY_LEN + 2,), dtype='int32')  # +2 for [CLS] and [SEP]
    query_input_mask = PaddingMask()(query_input)

    query_feature = feature_extractor([query_input, query_input_mask])

    query_attention = Dense(D_Q, activation='tanh')(query_feature)
    query_attention = Flatten()(Dense(1)(query_attention))
    query_attention_weight = Activation('softmax')(query_attention)
    query_rep = Dot((1, 1))([query_feature, query_attention_weight])

    query_encoder = tf.keras.Model(query_input, query_rep)

    # Article branch
    # # Sentence encoder
    sentence_input = Input(shape=(MAX_SENTENCE_LEN + 2,), dtype='int32')  # +2 for [CLS] and [SEP]
    sentence_input_mask = PaddingMask()(sentence_input)

    sentence_feature = feature_extractor([sentence_input, sentence_input_mask])

    sentence_attention = Dense(D_Q, activation='tanh')(sentence_feature)
    sentence_attention = Flatten()(Dense(1)(sentence_attention))
    sentence_attention_weight = Activation('softmax')(sentence_attention)
    sentence_rep = Dot((1, 1))([sentence_feature, sentence_attention_weight])

    sentence_encoder = tf.keras.Model(sentence_input, sentence_rep)

    # # Article encoder
    article_input = Input((MAX_NUM_SENTENCES + 2, MAX_SENTENCE_LEN + 2), dtype='int32')  # +2 for [CLS] and [SEP]
    sentences_rep = TimeDistributed(sentence_encoder)(article_input)

    article_attention = Dense(D_Q, activation='tanh')(sentences_rep)
    article_attention = Flatten()(Dense(1)(article_attention))
    article_attention_weight = Activation('softmax')(article_attention)
    article_rep = Dot((1, 1))([sentences_rep, article_attention_weight])

    article_encoder = tf.keras.Model(article_input, article_rep)

    # Scoring
    # +2 for [CLS] and [SEP]
    articles_input = [Input((MAX_NUM_SENTENCES + 2, MAX_SENTENCE_LEN + 2), dtype='int32') for _ in range(GROUP_SIZE)]
    articles_rep = [article_encoder(articles_input[_]) for _ in range(GROUP_SIZE)]
    logits = [dot([query_rep, a_rep], axes=-1) for a_rep in articles_rep]
    logits = concatenate(logits)
    logits = Activation(tf.keras.activations.softmax)(logits)

    model = tf.keras.Model([query_input] + articles_input, logits)

    return model, query_encoder, article_encoder


if __name__ == '__main__':
    create_model()
