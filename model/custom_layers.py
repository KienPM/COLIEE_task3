import tensorflow as tf
import tensorflow.keras.backend as K
from transformers import TFBertModel, BertConfig


class PaddingMask(tf.keras.layers.Layer):
    def __init__(self):
        super(PaddingMask, self).__init__()

    def call(self, inputs):
        return tf.cast(tf.math.equal(inputs, 0), tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape


class FeatureExtractor(tf.keras.layers.Layer):
    GET_CLS = 0
    GET_LAST_HIDDEN_STATE = 1
    GET_LAST_4_HIDDEN_STATES = 2

    def __init__(self, mode=GET_CLS):
        super(FeatureExtractor, self).__init__()
        if mode != self.GET_CLS:
            output_hidden_states = False
        else:
            output_hidden_states = True
        config = BertConfig.from_pretrained("nlpaueb/legal-bert-base-uncased",
                                            output_hidden_states=output_hidden_states)
        self.legal_bert = TFBertModel.from_pretrained("nlpaueb/legal-bert-base-uncased", config=config)
        self.legal_bert.layers[0].embeddings.trainable = False
        for layer in self.legal_bert.layers[0].encoder.layer.layers[:10]:
            layer.trainable = False

    def call(self, inputs):
        input_ids = inputs[0]
        input_masks_ids = inputs[1]
        self.result = self.legal_bert(input_ids, input_masks_ids)[0]
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
