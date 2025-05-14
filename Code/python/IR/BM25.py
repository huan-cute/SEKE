import os
import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from pre_generate import set_generation


def bm25_similarity(queried_file, query_file):
    # 生成被查询集
    queried_file = set_generation(queried_file)
    # 生成查询集
    query_file = set_generation(query_file)

    # 初始化BM25对象
    bm25_obj = BM25Okapi(queried_file)

    # 计算每个查询与被查询集的相似度
    sim = []
    for query in query_file:
        scores = bm25_obj.get_scores(query)
        sim.append(scores)

    # 转换为DataFrame
    sim_df = pd.DataFrame(sim)

    return sim_df
