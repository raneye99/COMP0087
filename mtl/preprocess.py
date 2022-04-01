import pandas as pd
import numpy as np

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