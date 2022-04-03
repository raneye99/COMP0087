from keras.models import Sequential, Model, Input
from keras.layers import Embedding, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras import layers
import tensorflow as tf
import datetime
from tensorboard.plugins.hparams import api as hp


class net:

    def mtl_model(self, num_classes, num_classes1, embed_num_dims, max_seq_len, vocab_size, embedd_matrix, gru_output_size=128, dropout=0.2, recurrent_dropout=0.2):

        ip = Input(shape=(500,)) #8164
        embedd_layer = Embedding(vocab_size,
                                embed_num_dims,
                                input_length = max_seq_len,
                                weights = [embedd_matrix],
                                trainable=False)(ip)
        bi = Bidirectional(GRU(units=gru_output_size,
                                    dropout=dropout,
                                    recurrent_dropout=recurrent_dropout))(embedd_layer)
        s = Dense(num_classes, activation='softmax', name='s')(bi)
        e1 = Dense(num_classes1, activation='softmax', name='e1')(bi)
        e2 = Dense(num_classes1, activation='softmax', name='e2')(bi)
        e3 = Dense(num_classes1, activation='softmax', name='e3')(bi)
        e4 = Dense(num_classes1, activation='softmax', name='e4')(bi)
        e5 = Dense(num_classes1, activation='softmax', name='e5')(bi)
        e6 = Dense(num_classes1, activation='softmax', name='e6')(bi)

        output = [s, e1, e2, e3, e4, e5, e6]
        model = Model(ip, output)
        model.summary()
        return model
    
    def mtl_five_sentiments_model(self, num_classes1, embed_num_dims, max_seq_len, vocab_size, embedd_matrix, gru_output_size=128, dropout=0.2, recurrent_dropout=0.2):

        ip = Input(shape=(500,)) #8164
        embedd_layer = Embedding(vocab_size,
                                embed_num_dims,
                                input_length = max_seq_len,
                                weights = [embedd_matrix],
                                trainable=False)(ip)
        bi = Bidirectional(GRU(units=gru_output_size,
                                    dropout=dropout,
                                    recurrent_dropout=recurrent_dropout))(embedd_layer)
        s = Dense(5, activation='softmax', name='s')(bi)
        e1 = Dense(num_classes1, activation='softmax', name='e1')(bi)
        e2 = Dense(num_classes1, activation='softmax', name='e2')(bi)
        e3 = Dense(num_classes1, activation='softmax', name='e3')(bi)
        e4 = Dense(num_classes1, activation='softmax', name='e4')(bi)
        e5 = Dense(num_classes1, activation='softmax', name='e5')(bi)
        e6 = Dense(num_classes1, activation='softmax', name='e6')(bi)

        output = [s, e1, e2, e3, e4, e5, e6]
        model = Model(ip, output)
        model.summary()
        return model

    def gru_one_task_sentiment(self,num_classes, embed_num_dims, max_seq_len, vocab_size, embedd_matrix, gru_output_size=128, dropout=0.2, recurrent_dropout=0.2):

        ip = Input(shape=(500,)) #8164
        embedd_layer = Embedding(vocab_size,
                                embed_num_dims,
                                input_length = max_seq_len,
                                weights = [embedd_matrix],
                                trainable=False)(ip)
        bi = Bidirectional(GRU(units=gru_output_size,
                                    dropout=dropout,
                                    recurrent_dropout=recurrent_dropout))(embedd_layer)
        s = Dense(num_classes, activation='softmax', name='s')(bi)

        output = [s]
        model = Model(ip, output)
        model.summary()
        return model

    def gru_one_task_sentiment_five(self, embed_num_dims, max_seq_len, vocab_size, embedd_matrix, gru_output_size=128, dropout=0.2, recurrent_dropout=0.2):

        ip = Input(shape=(500,)) #8164
        embedd_layer = Embedding(vocab_size,
                                embed_num_dims,
                                input_length = max_seq_len,
                                weights = [embedd_matrix],
                                trainable=False)(ip)
        bi = Bidirectional(GRU(units=gru_output_size,
                                    dropout=dropout,
                                    recurrent_dropout=recurrent_dropout))(embedd_layer)
        s = Dense(5, activation='softmax', name='s')(bi)

        output = [s]
        model = Model(ip, output)
        model.summary()
        return model

    def lstm_one_task_sentiment(self,num_classes, embed_num_dims, max_seq_len, vocab_size, embedd_matrix, gru_output_size=128, dropout=0.2, recurrent_dropout=0.2):

        ip = Input(shape=(500,)) #8164
        embedd_layer = Embedding(vocab_size,
                                embed_num_dims,
                                input_length = max_seq_len,
                                weights = [embedd_matrix],
                                trainable=False)(ip)
        bi = Bidirectional(LSTM(units=gru_output_size,
                                    dropout=dropout,
                                    recurrent_dropout=recurrent_dropout))(embedd_layer)
        s = Dense(num_classes, activation='softmax', name='s')(bi)

        output = [s]
        model = Model(ip, output)
        model.summary()
        return model

    def lstm_one_task_sentiment_five(self,num_classes, embed_num_dims, max_seq_len, vocab_size, embedd_matrix, gru_output_size=128, dropout=0.2, recurrent_dropout=0.2):

        ip = Input(shape=(500,)) #8164
        embedd_layer = Embedding(vocab_size,
                                embed_num_dims,
                                input_length = max_seq_len,
                                weights = [embedd_matrix],
                                trainable=False)(ip)
        bi = Bidirectional(LSTM(units=gru_output_size,
                                    dropout=dropout,
                                    recurrent_dropout=recurrent_dropout))(embedd_layer)
        s = Dense(num_classes, activation='softmax', name='s')(bi)

        output = [s]
        model = Model(ip, output)
        model.summary()
        return model

