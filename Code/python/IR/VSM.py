import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from pre_generate import set_generation
from gensim import corpora, models, similarities
from xlsxwriter import Workbook

def vsm_similarity(queried_file, query_file):
    # 生成被查询集
    queried_file = set_generation(queried_file)
    # 生成查询集
    query_file = set_generation(query_file)
    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_file)
    corpus = [dictionary.doc2bow(text) for text in queried_file]
    # 计算tfidf值
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]
    # 待检索的文档向量初始化一个相似度计算的对象
    corpus_sim = similarities.MatrixSimilarity(corpus_tfidf)
    # 查询集生成corpus和tfidf值
    query_corpus = [dictionary.doc2bow(text) for text in query_file]
    query_tfidf = tfidf_model[query_corpus]
    sim = pd.DataFrame(corpus_sim[query_tfidf])
    return sim







