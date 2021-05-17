import tensorflow as tf

class Transformer(object):
    
    
    def __init__(self, voca_size, emb_size, n_head=6):

        self.emb_size = emb_size  # same as d_model (in paper = 512)
        self.voca_size = voca_size
        self.n_head = n_head
        
        self.embedding = tf.keras.layers.Embedding(input_dim=voca_size, output_dim=emb_size)
    
    
    def _positional_encoding(self, sentence_vec):
        '''
        sentence_vec: tf.Tensor with shape (Batch, TimeStamp, Embedding dim)
        
        '''
        
        
    
    
    def multi_head_attention(self, q, k, v):
        
        Q = tf.keras.layers.Dense(self.emb_size * self.n_head)(q)
        K = tf.keras.layers.Dense(self.emb_size * self.n_head)(k)
        V = tf.keras.layers.Dense(self.emb_size * self.n_head)(v)
    
        Q = tf.reshape(Q, shape=(-1, self.voca_size, self.emb_size, self.n_head))
        K = tf.reshape(K, shape=(-1, self.voca_size, self.emb_size, self.n_head))
        V = tf.reshape(V, shape=(-1, self.voca_size, self.emb_size, self.n_head))
        
        
        QK = tf.matmul(Q, tf.transpose(K)) 
        QK_scaled /= (self.emb_size * 3) ** 0.5
        
        attention = tf.keras.layers.Softmax(QK_scaled)
        context_vec = tf.matmul(attention, V)
        
        return context_vec
    
    def encoding(self, xs):
        emb_vec = self.embedding(xs)  # (batch, n_voca, output_dim)
