import os
import re
from gensim import corpora, models, similarities
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

def set_generation(query_file):
    with open(query_file,"r",encoding="ISO-8859-1") as ft:
        lines_T = ft.readlines()
    setline=[]
    for line in lines_T:
        word = line.split(' ')
        word = [re.sub('\s','',i) for i in word]
        word = [i for i in word if len(i) > 0]
        setline.append(word)
    return setline