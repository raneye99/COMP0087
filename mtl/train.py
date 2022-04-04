
from keras.models import Sequential, Model, Input
from keras.layers import Embedding, Bidirectional, LSTM, GRU, Dense
from tensorflow.keras import layers
import tensorflow as tf
import datetime
from tensorboard.plugins.hparams import api as hp

from mtl.network import net

def train_model(vocab_size, embedd_matrix, model_name, data_x, data_y, val_x, val_y, saved_model_name, tensorboard,  loss_weights, batch_size=128, epochs=1, gru_output_size=128, dropout=0.2, recurrent_dropout=0.2):

    print('Training a '+model_name+' model!')

    if model_name == 'mtl':
        mod = net().mtl_model(vocab_size = vocab_size, embedd_matrix = embedd_matrix, gru_output_size=gru_output_size, dropout=dropout, recurrent_dropout=recurrent_dropout, num_classes = 7, num_classes1=2, embed_num_dims=300, max_seq_len=500)
        mod.compile(loss = {'s':'categorical_crossentropy',
                            'e1':'categorical_crossentropy',
                            'e2':'categorical_crossentropy',
                            'e3':'categorical_crossentropy',
                            'e4':'categorical_crossentropy',
                            'e5':'categorical_crossentropy',
                            'e6':'categorical_crossentropy'}, 
                    optimizer = 'adam',
                    metrics = {'s':'accuracy',
                            'e1': 'accuracy',
                            'e2': 'accuracy',
                            'e3': 'accuracy',
                            'e4': 'accuracy',
                            'e5': 'accuracy',
                            'e6': 'accuracy'},
                    loss_weights={'s':loss_weights[0], 
                    'e1':loss_weights[1],
                    'e2':loss_weights[2],
                    'e3':loss_weights[3],
                    'e4':loss_weights[4],
                    'e5':loss_weights[5],
                    'e6':loss_weights[6]})

        if tensorboard == True:
            log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1)
            hparams_callback = hp.KerasCallback(log_dir, {
                'num_relu_units': 512,
                'dropout': 0.2
            })

            hist = mod.fit(data_x, data_y, 
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(val_x,val_y),
                            #validation_data=(X_test_pad,(y_test, y_test1, y_test2, y_test3, y_test4, y_test5, y_test6)),
                            callbacks=[tensorboard_callback, hparams_callback])
        else:
            hist = mod.fit(data_x, data_y,  
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(val_x,val_y))
        mod.save(saved_model_name)
    
    if model_name == 'mtl_five_sentiments':
        mod = net().mtl_five_sentiments_model(num_classes1 = 2, embed_num_dims=300, max_seq_len=500, vocab_size = vocab_size, embedd_matrix = embedd_matrix, gru_output_size=gru_output_size, dropout=dropout, recurrent_dropout=recurrent_dropout, num_classes = 5)
        mod.compile(loss = {'s':'categorical_crossentropy',
                            'e1':'categorical_crossentropy',
                            'e2':'categorical_crossentropy',
                            'e3':'categorical_crossentropy',
                            'e4':'categorical_crossentropy',
                            'e5':'categorical_crossentropy',
                            'e6':'categorical_crossentropy'}, 
                    optimizer = 'adam',
                    metrics = {'s':'accuracy',
                            'e1': 'accuracy',
                            'e2': 'accuracy',
                            'e3': 'accuracy',
                            'e4': 'accuracy',
                            'e5': 'accuracy',
                            'e6': 'accuracy'},
                    loss_weights={'s':loss_weights[0], 
                    'e1':loss_weights[1],
                    'e2':loss_weights[2],
                    'e3':loss_weights[3],
                    'e4':loss_weights[4],
                    'e5':loss_weights[5],
                    'e6':loss_weights[6]})

        if tensorboard == True:
            log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1)
            hparams_callback = hp.KerasCallback(log_dir, {
                'num_relu_units': 512,
                'dropout': 0.2
            })

            hist = mod.fit(data_x, data_y, 
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(val_x,val_y),
                            callbacks=[tensorboard_callback, hparams_callback])
        else:
            hist = mod.fit(data_x, data_y, 
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_data=(val_x,val_y))

        mod.save(saved_model_name)
    
    elif model_name == 'gru_one_task_sentiment':

        mod = net().gru_one_task_sentiment(vocab_size = vocab_size, embedd_matrix = embedd_matrix, gru_output_size=gru_output_size, dropout=dropout, recurrent_dropout=recurrent_dropout, num_classes = 7, embed_num_dims=300, max_seq_len=500)
        mod.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        if tensorboard == True:
            log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1)
            hparams_callback = hp.KerasCallback(log_dir, {
                'num_relu_units': 512,
                'dropout': 0.2
                })
            hist = mod.fit(data_x, data_y, 
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(val_x, val_y),
                    callbacks=[tensorboard_callback, hparams_callback])  
        else:
            hist = mod.fit(data_x, data_y, 
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(val_x,val_y))  
        mod.save(saved_model_name)

    elif model_name == 'gru_one_task_sentiment_five':

        mod = net().gru_one_task_sentiment_five(vocab_size = vocab_size, embedd_matrix = embedd_matrix, gru_output_size=gru_output_size, dropout=dropout, recurrent_dropout=recurrent_dropout, embed_num_dims =300, max_seq_len=500, num_classes = 5)
        mod.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        if tensorboard == True:
            log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1)
            hparams_callback = hp.KerasCallback(log_dir, {
                'num_relu_units': 512,
                'dropout': 0.2
                })
            hist = mod.fit(data_x, data_y, 
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(val_x,val_y),
                    callbacks=[tensorboard_callback, hparams_callback])  
        else:
            hist = mod.fit(data_x, data_y, 
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(val_x,val_y))  
        mod.save(saved_model_name)

    elif model_name == 'lstm_one_task_sentiment':

        mod = net().lstm_one_task_sentiment(vocab_size = vocab_size, embedd_matrix = embedd_matrix, gru_output_size=gru_output_size, dropout=dropout, recurrent_dropout=recurrent_dropout, embed_num_dims = 300, max_seq_len=500, num_classes=7)
        mod.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        if tensorboard == True:
            log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1)
            hparams_callback = hp.KerasCallback(log_dir, {
                'num_relu_units': 512,
                'dropout': 0.2
            })
            hist = mod.fit(data_x, data_y, 
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(val_x,val_y),
                    callbacks=[tensorboard_callback, hparams_callback])  
        else:
            hist = mod.fit(data_x, data_y, 
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_data=(val_x,val_y))          
        mod.save(saved_model_name)
        
    elif model_name == 'lstm_one_task_sentiment_five':

        mod = net().lstm_one_task_sentiment_five(vocab_size = vocab_size, embedd_matrix = embedd_matrix, gru_output_size=gru_output_size, dropout=dropout, recurrent_dropout=recurrent_dropout, embed_num_dims = 300, max_seq_len=500, num_classes=5)
        mod.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        if tensorboard == True:
            log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1)
            hparams_callback = hp.KerasCallback(log_dir, {
                'num_relu_units': 512,
                'dropout': 0.2
            })
            hist = mod.fit(data_x, data_y, 
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(val_x,val_y),
                    callbacks=[tensorboard_callback, hparams_callback])  
        else:
            hist = mod.fit(data_x, data_y, 
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_data=(val_x,val_y))          
        mod.save(saved_model_name)
        
    