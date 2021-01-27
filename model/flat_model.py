""" Create by Ken at 2021 Jan 27 """
import tensorflow as tf
from tensorflow.keras.layers import *

INPUT_VOCAB_SIZE = 2235
MAX_SENTENCE_LEN = 150
MAX_ARTICLE_LEN = 1000
GROUP_SIZE = 101
D_EMBEDDING = 200
D_CNN = 512
D_Q = 200
DROPOUT_RATE = 0.2

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def log_hyper_parameters(logger):
    logger.info("-" * 10)
    logger.info("Hyper parameters")
    logger.info(f"MAX_SENTENCE_LEN: {MAX_SENTENCE_LEN}")
    logger.info(f"MAX_ARTICLE_LEN: {MAX_ARTICLE_LEN}")
    logger.info(f"INPUT_VOCAB_SIZE: {INPUT_VOCAB_SIZE}")
    logger.info(f"GROUP_SIZE: {GROUP_SIZE}")
    logger.info(f"D_EMBEDDING: {D_EMBEDDING}")
    logger.info(f"D_CNN: {D_CNN}")
    logger.info(f"D_Q: {D_Q}")
    logger.info("-" * 10)


def create_model(embedding_matrix=None, with_triplet_loss=False):
    """
    Return: Model, Query encoder, Article encoder
    """
    if embedding_matrix is not None:
        embedding_layer = Embedding(INPUT_VOCAB_SIZE, D_EMBEDDING, weights=[embedding_matrix], trainable=True)
    else:
        embedding_layer = Embedding(INPUT_VOCAB_SIZE, D_EMBEDDING, trainable=True)

    # Query branch
    query_input = Input((MAX_SENTENCE_LEN,), dtype='int32')

    embedded_sequences_query = embedding_layer(query_input)
    embedded_sequences_article = Dropout(DROPOUT_RATE)(embedded_sequences_query)

    query_cnn = Convolution1D(filters=D_CNN, kernel_size=3, padding='same', activation='relu', strides=1)(
        embedded_sequences_article
    )
    query_cnn = Dropout(DROPOUT_RATE)(query_cnn)
    query_cnn = tf.keras.layers.LayerNormalization(epsilon=1e-6)(query_cnn)

    attention_query = Dense(D_Q, activation='tanh')(query_cnn)
    attention_query = Flatten()(Dense(1)(attention_query))
    attention_weight_query = Activation('softmax')(attention_query)
    query_rep = tf.keras.layers.Dot((1, 1))([query_cnn, attention_weight_query])

    query_encoder = tf.keras.Model(query_input, query_rep)

    # Article branch
    article_input = Input(shape=(MAX_ARTICLE_LEN,), dtype='int32')

    embedded_sequences_article = embedding_layer(article_input)
    embedded_sequences_article = Dropout(DROPOUT_RATE)(embedded_sequences_article)

    article_cnn = Convolution1D(filters=D_CNN, kernel_size=3, padding='same', activation='relu', strides=1)(
        embedded_sequences_article
    )
    article_cnn = Dropout(DROPOUT_RATE)(article_cnn)
    article_cnn = tf.keras.layers.LayerNormalization(epsilon=1e-6)(article_cnn)

    attention_article = Dense(D_Q, activation='tanh')(article_cnn)
    attention_article = Flatten()(Dense(1)(attention_article))
    attention_weight_article = Activation('softmax')(attention_article)
    article_rep = tf.keras.layers.Dot((1, 1))([article_cnn, attention_weight_article])

    article_encoder = tf.keras.Model(article_input, article_rep)

    # Scoring
    articles_input = [Input((MAX_ARTICLE_LEN,), dtype=tf.int32) for _ in range(GROUP_SIZE)]
    articles_rep = [article_encoder(articles_input[_]) for _ in range(GROUP_SIZE)]
    logits = [tf.keras.layers.dot([query_rep, article_rep], axes=-1) for article_rep in articles_rep]
    logits = tf.keras.layers.concatenate(logits)
    logits = tf.keras.layers.Activation(tf.keras.activations.softmax)(logits)

    model = tf.keras.Model([query_input] + articles_input, logits)

    return model, query_encoder, article_encoder
