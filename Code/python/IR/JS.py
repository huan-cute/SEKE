import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from pre_generate import set_generation
from gensim import corpora, models, similarities
from xlsxwriter import Workbook
import scipy.stats

# JS散度相似度计算
def JS_similarity(queried_file, query_file, output_fname=None):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)

    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_line + query_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]
    corpus2 = [dictionary.doc2bow(text) for text in query_line]
    A_matrix = np.zeros((len(queried_line), len(dictionary)))
    B_matrix = np.zeros((len(query_line), len(dictionary)))

    row = 0
    for document in corpus:
        for word_id, freq in document:
            A_matrix[row][word_id] = freq
        row = row + 1

    row = 0
    for document in corpus2:
        for word_id, freq in document:
            B_matrix[row][word_id] = freq
        row = row + 1

    sum_matrix = np.sum(np.vstack((A_matrix, B_matrix)), axis=0)
    probability_A = A_matrix / sum_matrix
    probability_B = B_matrix / sum_matrix

    sim = JS_Sim(probability_A, probability_B)

    if output_fname is not None:
        sim.to_excel(output_fname)
    return sim

def JS_Sim(A_set, B_set) -> pd.DataFrame:
    df = pd.DataFrame(index=range(len(B_set)), columns=range(len(A_set)))
    # 开始计算JS相似度
    for row in range(len(B_set)):
        for column in range(len(A_set)):
            df.iloc[[row], [column]] = JS_divergence(B_set[row], A_set[column])  # B_set为查询集，所以放前面
    return df

def JS_divergence(p, q):
    M = (p + q) / 2
    pk = np.asarray(p)
    pk2 = np.asarray(q)
    a = 0
    b = 0
    if (np.sum(pk, axis=0, keepdims=True) != 0):
        a = 0.5 * scipy.stats.entropy(p, M)
    if (np.sum(pk2, axis=0, keepdims=True) != 0):
        b = 0.5 * scipy.stats.entropy(q, M)
    return a + b  # 选用自然对数
