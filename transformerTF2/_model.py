import numpy as np
import tensorflow as tf

from .encoder import Encoder
from .decoder import Decoder


class Transformer(tf.keras.Model):

    def __init__(self, num_layer, d_model, n_heads, dff, input_voca_size, target_voca_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__(self)

        self.tokenizer = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_voca_size)

    def call(self,inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding-mask):

        enc_output = self.tokenizer(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights

    def _positional_encoding(self, sentence_vec):
        '''

        PE(pos, 2i) = sin(pos / 10000^{2i/d_model})
        PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})


        sentence_vec: tf.Tensor with shape (Batch, TimeStamp, Embedding dim)

        '''

        # Given shape [Batch, TimeStamp, Embedding ] dimensions.
        B, T, E = sentence_vec.shape


        pos_ind = tf.expand_dims(tf.range(T), axis=0)  # (1, T)
        pos_ind = tf.tile(pos_ind, multiples=[B, 1])

        pos_enc = [[pos / np.power(10000, 2 * (i // 2) / self.d_model)
                  for i in range(self.d_model)]  # [1.0 ~ approx 9999]
                  for pos in range(T)]
        pos_enc = np.array(pos_enc)  # (T, E(=model_d))


        # Apply cosine (odds), or sin (even)  (sinusodial embedding)
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])  # dim i+2

        pe = tf.convert_to_tensor(pos_enc)  # (T, E)

        return pe


    def multi_head_attention(self, q, k, v):
        Q = tf.keras.layers.Dense(self.d_model * self.n_head)(q)
        K = tf.keras.layers.Dense(self.d_model * self.n_head)(k)
        V = tf.keras.layers.Dense(self.d_model * self.n_head)(v)

        Q = tf.reshape(Q, shape=(-1, self.voca_size, self.d_model, self.n_head))
        K = tf.reshape(K, shape=(-1, self.voca_size, self.d_model, self.n_head))
        V = tf.reshape(V, shape=(-1, self.voca_size, self.d_model, self.n_head))

        QK = tf.matmul(Q, tf.transpose(K))
        QK_scaled /= (self.emb_size * 3) ** 0.5

        attention = tf.keras.layers.Softmax(QK_scaled)
        context_vec = tf.matmul(attention, V)

        return context_vec

    def encoding(self, xs):
        emb_vec = self.embedding(xs)  # (batch, n_voca, output_dim)

        pe = self._positional_encoding(emb_vec)
        return tf.add(emb_vec + pe)

