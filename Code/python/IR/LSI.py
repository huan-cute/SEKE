import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from pre_generate import set_generation
from gensim import corpora, models, similarities
from xlsxwriter import Workbook


# LSI相似度计算
def lsi_similarity(queried_file, query_file, output_fname=None):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)

    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]

    # 计算tfidf值
    tfidf_model = models.TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]

    # 生成lsi主题
    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary)
    corpus_lsi = lsi_model[corpus_tfidf]

    # 待检索的文档向量初始化一个相似度计算的对象
    corpus_sim = similarities.MatrixSimilarity(corpus_lsi)

    # 查询集生成corpus和tfidf值
    query_corpus = [dictionary.doc2bow(text) for text in query_line]  # 在每句话中每个词语出现的频率
    query_tfidf = tfidf_model[query_corpus]
    query_lsi = lsi_model[query_tfidf]
    sim = pd.DataFrame(corpus_sim[query_lsi])

    if output_fname is not None:
        sim.to_excel(output_fname)
    return sim
