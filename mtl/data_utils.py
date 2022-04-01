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


#add mtl to sys.path if its not there already so we can run file from command line/terminal without being in package directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)


def download_data(data = 'MOSEI_UMONS', outdir = 'data'):
    '''
    Download a file from url and place in data folder

    Note for our purposes, we will be using text mosei data located here: https://drive.google.com/file/d/1tcVYIMcZdlDzGuJvnMtbMchKIK9ulW1P/view

    Args:
    --------------
    fname: str or None, default None
        name of  to write files to, if none will write to datasets directory in mtl package

    '''
    #make data directory if doesn't exist in path folder
    os.makedirs(outdir, exist_ok=True)

    if data == 'MOSEI_UMONS':

        # #url to download from
        # base_url = "https://drive.google.com/file/d/1tcVYIMcZdlDzGuJvnMtbMchKIK9ulW1P/view?usp=sharing"

        # #split url to get file name
        # filename = base_url.split('/')[-1]

        # #download
        # req = requests.get(base_url)

        # #extract
        # zfile = zipfile.ZipFile(BytesIO(req.content))
        # zfile.extractall(outdir)
        file_id = '1tcVYIMcZdlDzGuJvnMtbMchKIK9ulW1P'
        url = "https://docs.google.com/uc?export=download?"
        session = requests.Session()
        response = session.get(url, params = {'id': id}, stream = True)

        #download meld and mosei zip files for data (mosei is from CMU, meld is friends episodes)
        os.system("!file=1tcVYIMcZdlDzGuJvnMtbMchKIK9ulW1P && wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${file} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=\"${file} -O data/data.zip && rm -rf /tmp/cookies.txt")

        os.system("!unzip data/data.zip -d data/mosei")
    
    else:
        print('unable to download, dataset is unknown')
        
    
