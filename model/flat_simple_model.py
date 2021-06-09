""" Create by Ken at 2021 Jan 27 """
import tensorflow as tf
from tensorflow.keras.layers import *
from custom_layers import FeatureExtractor, PaddingMask

MAX_QUERY_LEN = 150
MAX_ARTICLE_LEN = 500
D_Q = 200

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def log_hyper_parameters(logger):
    logger.info("-" * 10)
    logger.info("Hyper parameters")
    logger.info(f"MAX_SENTENCE_LEN: {MAX_QUERY_LEN}")
    logger.info(f"MAX_ARTICLE_LEN: {MAX_ARTICLE_LEN}")
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
    article_input = Input(shape=(MAX_ARTICLE_LEN + 2,), dtype='int32')  # +2 for [CLS] and [SEP]
    article_input_mask = PaddingMask()(article_input)

    article_feature = feature_extractor([article_input, article_input_mask])

    article_attention = Dense(D_Q, activation='tanh')(article_feature)
    article_attention = Flatten()(Dense(1)(article_attention))
    article_attention_weight = Activation('softmax')(article_attention)
    article_rep = tf.keras.layers.Dot((1, 1))([article_feature, article_attention_weight])

    article_encoder = tf.keras.Model(article_input, article_rep)

    # Scoring
    logits = tf.keras.layers.dot([query_rep, article_rep], axes=-1)
    logits = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(logits)

    model = tf.keras.Model([query_input, article_input], logits)

    return model, query_encoder, article_encoder


if __name__ == '__main__':
    model, q_e, a_e = create_model()
    q_e.summary()
    a_e.summary()
