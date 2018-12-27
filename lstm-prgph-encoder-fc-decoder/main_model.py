from tensorflow.keras import Model
import encoder
import classifier


class main_model(Model):
    """
    desc. : This model does the task of paragraph transposition. An abstract schema of this model would contain two
            input sequences, an encoder model and a classifier to determine the inputs order.
    """

    def __init__(self, name , embedding_table , hidden_len , encoder_dropout_rate , classifier_dropout_rate):
        super(main_model, self).__init__()
        self._name = name
        self.encoder = encoder(self._name+'/LSTM_encoder', embedding_table, hidden_len, encoder_dropout_rate)
        self._classifier = classifier(self._name+'/FC_classifier')


    def call(self, inputs):
        assert isinstance(inputs , list)
        x = encoder(inputs[0])
        y= encoder (inputs[1])

        output = self._classifier(x , y)

        return output


