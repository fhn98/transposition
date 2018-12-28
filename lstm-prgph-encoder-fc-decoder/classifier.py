import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np


class Classifier(Model):
    
    def __init__(self, name, num_classes = 1, hidden_dim = 64, dropout = 0.2):
        """
        This classifier determines if paragraphs are forward or backward
        :param name: name of the classifier
        :param num_classes: dimension of output
        :param hidden_dim: dimension of hidden layer
        :param dropout: dropout value
        """
        
        super(Classifier, self).__init__()
        self._name = name
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout = layers.Dropout(dropout)
        self.layer1 = layers.Dense(self.hidden_dim, activation = 'relu') 
        self.layer2 = layers.Dense(self.num_classes, activation = 'tanh')
    
    @property
    def variables(self):
        return [x.variables for x in self._layers]
    
    @property
    def trainable_variables(self):
        return [var for var in self._variables if self._trainable_variables[var]]
    
    
    def call(self, inputs, is_training = True):
        y = self.layer1(inputs)
        if is_training:
            y = self.dropout(y)
        return self.layer2(y)