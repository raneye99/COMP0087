from copyreg import pickle
import pandas as pd
import numpy as np
import pickle

# text preprocessing
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import re
import urllib.request
import zipfile
import os

# preparing input to our model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

def clean_text(data):
    
    # remove hashtags and @usernames
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)
    
    # tekenization using nltk
    data = word_tokenize(data)
    
    return data

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix

def data_processing(five_sentiments = True, num_classes = 7, num_classes1 = 2, embed_num_dims = 300, max_seq_len = 500):

    class_names = ['0.0', '1.0', '2.0', '3.0', '4.0', '5.0', '6.0']
    class_names1 = ['0.0', '1.0']

    #load in data
    #train
    text_tr = pickle.load(open('data/mosei/MOSEI/train_sentences.p',"rb"))
    emotion_tr = pickle.load(open('data/mosei/MOSEI/train_emotion.p', "rb"))
    sentiment_tr = pickle.load(open('data/mosei/MOSEI/train_sentiment.p', "rb"))
    #change sentimement to be int
    sentiment_tr = {k: float(np.round(v)+3) for k,v in sentiment_tr.items()}


    #valid
    text_val = pickle.load(open('data/mosei/MOSEI/valid_sentences.p',"rb"))
    emotion_val = pickle.load(open('data/mosei/MOSEI/valid_emotion.p', "rb"))
    sentiment_val = pickle.load(open('data/mosei/MOSEI/valid_sentiment.p', "rb"))
    #change sentiement to be int
    sentiment_val = {k: float(np.round(v)+3) for k,v in sentiment_val.items()}

    #test
    text_test = pickle.load(open('data/mosei/MOSEI/test_sentences.p',"rb"))
    emotion_test = pickle.load(open('data/mosei/MOSEI/test_emotion.p', "rb"))
    sentiment_test= pickle.load(open('data/mosei/MOSEI/test_sentiment.p', "rb"))
    #change sentiement to be int
    sentiment_test = {k: float(np.round(v)+3) for k,v in sentiment_test.items()}

    col_name = ['sentences','emotion','sentiment']
    
    #create pandas dataframe for train validation and test set
    df_train = pd.DataFrame.from_dict([text_tr, emotion_tr, sentiment_tr]).T
    df_train.columns = col_name

    df_val= pd.DataFrame.from_dict([text_val, emotion_val, sentiment_val]).T
    df_val.columns = col_name

    df_test= pd.DataFrame.from_dict([text_test, emotion_test, sentiment_test]).T
    df_test.columns = col_name

    emo_cols = ['emotion_1', 'emotion_2', 'emotion_3', 'emotion_4', 'emotion_5', 'emotion_6']
    emo_train = pd.Series(df_train['emotion']).apply(lambda x: pd.Series(x))
    emo_train.columns = emo_cols

    emo_val = pd.Series(df_val['emotion']).apply(lambda x: pd.Series(x))
    emo_val.columns = emo_cols

    emo_test = pd.Series(df_test['emotion']).apply(lambda x: pd.Series(x))
    emo_test.columns = emo_cols

    #detect presense of emotion
    emo_train = emo_train.where(emo_train==0, 1)
    emo_val = emo_val.where(emo_train==0, 1)
    emo_test = emo_test.where(emo_test==0,1)

    data_train = df_train.join(emo_train)
    data_val = df_val.join(emo_val)
    data_test = df_test.join(emo_test)

    if five_sentiments:
        data_train = data_train.drop(data_train[data_train.sentiment == 0.0].index)
        data_train = data_train.drop(data_train[data_train.sentiment == 6.0].index)
        
        data_test = data_test.drop(data_test[data_test.sentiment == 0.0].index)
        data_test = data_test.drop(data_test[data_test.sentiment == 6.0].index)

        data_val = data_val.drop(data_val[data_val.sentiment == 0.0].index)
        data_val = data_val.drop(data_val[data_val.sentiment == 6.0].index)

    #even out the dataset by only allowing 3000 sentences with sentiment 3.0
    threes = data_train[data_train.sentiment == 3.0]
    threes = threes.iloc[:3000]
    non_threes = data_train[data_train.sentiment != 3.0]
    data_train = threes.append(non_threes, ignore_index=True)
    data_train = shuffle(data_train)

    X_train = data_train.sentences
    X_test = data_test.sentences
    X_val = data_val.sentences

    y_train = data_train.sentiment
    y_test = data_test.sentiment
    y_train1 = data_train.emotion_1
    y_test1 = data_test.emotion_1
    y_train2 = data_train.emotion_2
    y_test2 = data_test.emotion_2
    y_train3 = data_train.emotion_3
    y_test3 = data_test.emotion_3
    y_train4 = data_train.emotion_4
    y_test4 = data_test.emotion_4
    y_train5 = data_train.emotion_5
    y_test5 = data_test.emotion_5
    y_train6 = data_train.emotion_6
    y_test6 = data_test.emotion_6

    y_val = data_val.sentiment
    y_val1 = data_val.emotion_1
    y_val2 = data_val.emotion_2
    y_val3 = data_val.emotion_3
    y_val4 = data_val.emotion_4
    y_val5 = data_val.emotion_5
    y_val6 = data_val.emotion_6
    
    data = data_train.append(data_test, ignore_index=True) #VALIDATION???

    print('Training data sentiment statistics')
    print(data_train.sentiment.value_counts())
    print('Test data sentiment statistics')
    print(data_test.sentiment.value_counts())
    print('Validation data sentiment statistics')
    print(data_val.sentiment.value_counts())
    data.head(6)

    texts = [' '.join(clean_text(str(text))) for text in data.sentences]
    texts_train = [' '.join(clean_text(str(text))) for text in X_train]
    texts_test = [' '.join(clean_text(str(text))) for text in X_test]
    texts_val = [' '.join(clean_text(str(text))) for text in X_val]

    print(texts_train[92])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    sequence_train = tokenizer.texts_to_sequences(texts_train)
    sequence_test = tokenizer.texts_to_sequences(texts_test)
    sequence_val = tokenizer.texts_to_sequences(texts_val)

    index_of_words = tokenizer.word_index

    # vacab size is number of unique words + reserved 0 index for padding
    vocab_size = len(index_of_words) + 1

    print('Number of unique words: {}'.format(len(index_of_words)))

    X_train_pad = pad_sequences(sequence_train, maxlen = max_seq_len)
    X_test_pad = pad_sequences(sequence_test, maxlen = max_seq_len)
    X_val_pad = pad_sequences(sequence_val, maxlen = max_seq_len)

    if five_sentiments == True:
        encoding = {
            '1.0': 0,
            '2.0': 1,
            '3.0': 2,
            '4.0': 3,
            '5.0': 4
        }

        encoding1 = {
            '0.0': 0,
            '1.0': 1
        }
    else:
        encoding = {
            '0.0': 0,
            '1.0': 1,
            '2.0': 2,
            '3.0': 3,
            '4.0': 4,
            '5.0': 5,
            '6.0': 6
        }

        encoding1 = {
            '0.0': 0,
            '1.0': 1
        }
    

    print(encoding)

    y_train1 = [encoding1[str(x)] for x in data_train.emotion_1]
    y_test1 = [encoding1[str(x)] for x in data_test.emotion_1]
    y_train2 = [encoding1[str(x)] for x in data_train.emotion_2]
    y_test2 = [encoding1[str(x)] for x in data_test.emotion_2]
    y_train3 = [encoding1[str(x)] for x in data_train.emotion_3]
    y_test3 = [encoding1[str(x)] for x in data_test.emotion_3]
    y_train4 = [encoding1[str(x)] for x in data_train.emotion_4]
    y_test4 = [encoding1[str(x)] for x in data_test.emotion_4]
    y_train5 = [encoding1[str(x)] for x in data_train.emotion_5]
    y_test5 = [encoding1[str(x)] for x in data_test.emotion_5]
    y_train6 = [encoding1[str(x)] for x in data_train.emotion_6]
    y_test6 = [encoding1[str(x)] for x in data_test.emotion_6]
    y_train = [encoding[str(x)] for x in data_train.sentiment]
    y_test = [encoding[str(x)] for x in data_test.sentiment]

    y_val = [encoding[str(x)] for x in data_val.sentiment]
    y_val1 = [encoding1[str(x)] for x in data_val.emotion_1]
    y_val2 = [encoding1[str(x)] for x in data_val.emotion_2]
    y_val3 = [encoding1[str(x)] for x in data_val.emotion_3]
    y_val4 = [encoding1[str(x)] for x in data_val.emotion_4]
    y_val5 = [encoding1[str(x)] for x in data_val.emotion_5]
    y_val6 = [encoding1[str(x)] for x in data_val.emotion_6]

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    y_train1 = to_categorical(y_train1)
    y_test1 = to_categorical(y_test1)
    y_train2 = to_categorical(y_train2)
    y_test2 = to_categorical(y_test2)
    y_train3 = to_categorical(y_train3)
    y_test3 = to_categorical(y_test3)
    y_train4 = to_categorical(y_train4)
    y_test4 = to_categorical(y_test4)
    y_train5 = to_categorical(y_train5)
    y_test5 = to_categorical(y_test5)
    y_train6 = to_categorical(y_train6)
    y_test6 = to_categorical(y_test6)

    y_val = to_categorical(y_val)
    y_val1 = to_categorical(y_val1)
    y_val2 = to_categorical(y_val2)
    y_val3 = to_categorical(y_val3)
    y_val4 = to_categorical(y_val4)
    y_val5 = to_categorical(y_val5)
    y_val6 = to_categorical(y_val6)

    fname = 'embeddings/wiki-news-300d-1M.vec'

    if not os.path.isfile(fname):
        print('Downloading word vectors...')
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip',
                                'wiki-news-300d-1M.vec.zip')
        print('Unzipping...')
        with zipfile.ZipFile('wiki-news-300d-1M.vec.zip', 'r') as zip_ref:
            zip_ref.extractall('embeddings')
        print('done.')
        
        os.remove('wiki-news-300d-1M.vec.zip')


    embedd_matrix = create_embedding_matrix(fname, index_of_words, embed_num_dims)

    new_words = 0

    for word in index_of_words:
        entry = embedd_matrix[index_of_words[word]]
        if all(v == 0 for v in entry):
            new_words = new_words + 1

    print('Words found in wiki vocab: ' + str(len(index_of_words) - new_words))
    print('New words found: ' + str(new_words))

    return [X_train, X_test, X_train_pad, X_test_pad, y_train, y_test, y_train1, y_test1, 
        y_train2, y_test2, y_train3, y_test3, y_train4, y_test4, 
        y_train5, y_test5, y_train6, y_test6, embedd_matrix, vocab_size, X_val, X_val_pad, y_val,
        y_val1, y_val2, y_val3, y_val4, y_val5, y_val6, data_test]