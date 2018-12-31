import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np


class CNNEncoder2(Model):

    def __init__(self, name, embedding_table, hidden_len, num_layer=1, kernel_size=4, filters=1, padding='SAME',
                 pool_size=3, dropout=0.2):
        """
        Encoder of paragraph.
        Args:
            name: the name of the classifier
            embeding_table: the table of embeddingd
            hidden_len: the dimensions of hidden layers
            num_layer: the number of layers in CNN
            kernel_size: the length of the 1D convolution window
            filters: the number of output filters in the convolution
            padding: value of padding
            pool_size: size of the max pooling windows
            dropout: the dropout value
        """

        super(CNNEncoder2, self).__init__()
        self._name = name

        self._vocab_size = embedding_table.shape[0]
        self._embedding_len = embedding_table.shape[1]
        self._hidden_len = hidden_len
        embedding_initializer = keras.initializers.constant(embedding_table)
        self._embeddings = Embedding(self._vocab_size, self._embedding_len, embedding_initializer,
                                     trainable=False, mask_zero=True)

        self.num_layer = num_layer
        self.kernel_size = kernel_size
        self.filters = filters
        self.dropout = layers.Dropout(dropout)
        self.conv = layers.Conv1D(kernel_size=kernel_size, filters=filters, padding=padding,
                                  name=name + '/Conv1D')
        self.pool = layers.MaxPool1D(pool_size=pool_size, name=name + '/MaxPool1D')
        self.global_pool = layers.GlobalAveragePooling1D(name=name + '/GlobalAvePool1D')

    @property
    def variables(self):
        return [x.variables for x in self._layers]

    @property
    def trainable_variables(self):
        return [var for var in self._variables if self._trainable_variables[var]]

    def call(self, inputs, is_training=True):
        shape = inputs.shape
        y = inputs.reshape(shape[0], shape[1] * shape[2], shape[3])

        for i in range(0, self.num_layer):
            y = self.conv(y)
            y = self.pool(y)
            if is_training:
                y = self.dropout(y)

        return self.global_pool(y)