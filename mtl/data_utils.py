#import libraries
import pickle
from collections import defaultdict
from urllib.request import urlopen
import numpy as np
import pandas as pd
import os
import io
import h5py
import sys
import requests
import zipfile
import inspect
from io import BytesIO


#git current directory and working directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

# if currentdir not in sys.path:
#     sys.path.extend([currentdir])

def download():
    print("WIP")
        
    
