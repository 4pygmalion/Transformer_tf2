import tensorflow as tf
from .attention import MultiHeadAttention

__all__ = ['DecoderLayer', 'Decoder']

class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, n_heads, d_model, dff, dropout_rate=0.1):
        super(Decoder, self).__init__()

        # block 1
        self.mha1 = MultiHeadAttention()
        self.norm1 = tf.keras.layers.Normalization(1e-6)
        self.dropout1 = tf.keras.layers.dropout(dropout_rate)

        # block 2
        self.mha2 = MultiHeadAttention()
        self.norm2 = tf.keras.layers.Normalization(1e-6)
        self.dropout2 = tf.keras.layers.dropout(dropout_rate)

        # block 3
        self.ffn = tf.keras.layers.Dense(dff)
        self.norm3 = tf.keras.layers.Normalization(1e-6)
        self.dropout3 = tf.keras.layers.dropout(dropout_rate)


    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask=None):

        # block 1
        att_output, att_weight_block1 = self.mha1(x, x, x, look_ahead_mask)
        att_output = self.dropout1(att_output, training=training)
        out = self.norm1(x + att_output)


        # block 2
        att_output2, att_weight_block2 = self.mha2(encoder_output, encoder_output, out, padding_mask)
        att_output2 = self.dropout2(att_output2, training=training)
        out2 = self.norm2(att_output2 + out)

        # block 3
        ffn_out = self.ffn(out)
        ffn_out = self.dropout3(x, training=training)
        out = self.norm2(ffn_out + out2)

        return out, att_weight_block1, att_weight_block2



class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}

    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)

      attention_weights[f'decoder_layer{i+1}_block1'] = block1
      attention_weights[f'decoder_layer{i+1}_block2'] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights