import numpy as np
import tensorflow as tf


__all__ = ['positional_encoding', 'Attention', 'MultiHeadAttention']


def positional_encoding(sentence_length, d_model):
    '''PE(pos, 2i) = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

    Parameters
    ----------
    d_model = int, the number of embedding vector size
    sentence_length: int,


    Returns
    ------
    positional encoding: tf.tensor

    References
    -----
    https://www.tensorflow.org/text/tutorials/transformer#positional_encoding
    '''

    # Given shape
    encoded_vec = np.array([pos / np.power(10000, 2*i/d_model)
                         for pos in range(sentence_length)
                         for i in range(d_model)])

    # Apply cosine (odds), or sin (even)  (sinusodial embedding)
    encoded_vec[0::2] = np.sin(encoded_vec[0::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])  # dim i+2

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, d_model]))


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, n_heads=6, d_model=300):
        '''

        Parameters
        ----------
        n_heads: int:
            default 6
        d_model: positive integer. the number of vector size to embedded.
            (defualt 300)
        '''

        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.depth = d_model // self.n_heads

        self.wq = tf.keras.layers.Dense(d_model) # (batch, lenght(time), d_model(embedding)
        self.wk= tf.keras.layers.Dense(d_model)  # (batch, lenght(time), d_model(embedding)
        self.wv = tf.keras.layers.Dense(d_model) # (batch, lenght(time), d_model(embedding)
        self.dense = tf.keras.layers.Dense(d_model)

        if d_model % self.n_heads != 0:
            raise ValueError("The number of softembedding vec must be divided by n_head")


    def split_heads(self, x, batch_size):
        ''' Split the last dimenesion (heads, depth) -> (head, head, length, depth)

        Parameters
        ----------
        x: tf.Tensor
        batch_size:

        Returns
        -------
        tf. Tensor

        '''


        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth)) # batch_size, -1, self.n_heads, self.d_model
        return tf.transpose(x, perm=(0, 2, 1, 3))  #  batch_size, self.n_heads, -1, self.d_model


    def call(self, q, k ,v, mask=None):
        # Shape of v, k, q: (batch, seq_len(time), vec)

        batch_size = tf.shape(q)[0] # (batch)

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch, head, seq_len, depth)
        k = self.split_heads(k, batch_size)  # (batch, head, seq_len, depth)
        v = self.split_heads(v, batch_size)  # (batch, head, seq_len, depth)

        # (batch, heads, seq_len, depth), (batch, heads, seq_len, seq_len)
        scaled_att, att_weight = self.scaled_dot_product_attention(q, k, v, mask)

        # (batch, seq_len, heads, depth)
        scaled_att = tf.transpose(scaled_att, perm=[0, 2, 1, 3])
        concat_att = tf.reshape(scaled_att, shape=(batch_size, -1, self.d_model))  # (batch, seq_len, d_model)
        output = self.dense(concat_att)  # (batch, seq_len, d_model)
        return output, att_weight

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        '''

        Parameters
        ----------
        q, k, v: tf.Tensor. shape=(batch, heads, seq_len_k, d_model)
        mask : bool

        Returns
        -------
        tuple: context vector, attention weight
        '''

        d_k = tf.cast(tf.shape(k)[-1], tf.float32) # for avoiding InvalidArgumentError

        # (batch, head, seq_len_k, d_model) x (batch, head, d_model, seq_len_k)
        # equivalent to qk = tf.linalg.matmul(q, k, transpose_b=True)
        qk = tf.linalg.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2]))   # (batch, head, seq_len_k, seq_len_k)
        scaled_qk = qk / tf.math.sqrt(d_k)  # (batch, head, seq_len_k, seq_len_k)

        att_weight = tf.nn.softmax(scaled_qk, axis=-1) # (... seq_len_k, seq_len_k) -> (batch, head, seq_len_k, seq_len_k)

        if mask is not None:
            att_weight += (mask * -1e9)  # adding extremely minimal value

        # (batch, head, seq_len_k, seq_len_k) X (batch, head, seq_len_k, d_model)  => (batch, head, seq_len_k, d_model)
        return tf.matmul(att_weight, v), att_weight


