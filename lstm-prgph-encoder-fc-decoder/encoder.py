from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import *

class encoder(Model):
    def __init__(self, name, embedding_table, hidden_len, dropout_rate):
        super(encoder, self).__init__()

        self._name = name
        self._vocab_size = embedding_table.shape[0]
        self._embedding_len = embedding_table.shape[1]
        self._hidden_len = hidden_len
        self.dropout_rate = dropout_rate


        embedding_initializer = keras.initializers.constant(embedding_table)
        self._embeddings = Embedding(self._vocab_size , self._embedding_len , embedding_initializer ,
                                     trainable=False , mask_zero=True , name=self._name+'/embedding')
        self._LSTM = LSTM(self._hidden_len , name=self._name+'/LSTM')




    def call(self , seq):
        x = self._embeddings(seq)
        output = self._LSTM(x)

        return output

