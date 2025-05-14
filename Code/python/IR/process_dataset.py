import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from gensim import similarities
from VSM import vsm_similarity
from LSI import lsi_similarity
from BM25 import bm25_similarity
from JS import JS_similarity
from LDA import lda_similarity
def process_dataset(dataset_name, output_excel_path):
    # 计算相似度矩阵
    sim = lsi_similarity('../dataset_output/cc/' + dataset_name + '.txt', '../dataset_output/uc/' + dataset_name + '.txt')

    uc_dir = os.path.join('..', 'dataset', 'uc', dataset_name)
    cc_dir = os.path.join('..', 'dataset', 'cc', dataset_name)
    true_set_file = os.path.join('..', 'dataset', 'true_set', dataset_name + '.txt')

    uc_names = os.listdir(uc_dir)
    cc_names = os.listdir(cc_dir)

    # 展平相似度矩阵
    sim_flat = sim.values.flatten()

    # 读取真集
    with open(true_set_file, 'r', encoding='ISO-8859-1') as tf:
        lines = tf.readlines()
        relevant_pairs = [line.strip() for line in lines]

    true_set = [0] * (len(uc_names) * len(cc_names))

    for i, query_id in enumerate(uc_names):
        if dataset_name == 'eTour':
            query_id = query_id.replace(".TXT", "")
        else:
            query_id = query_id.replace(".txt", "")
        for j, document_id in enumerate(cc_names):
            pair = f"{query_id} {document_id}"
            if pair in relevant_pairs:
                index = i * len(cc_names) + j
                true_set[index] = 1

    # 准备存储结果的列表
    results = []
    for percentile in range(1, 101):  # 从1%到100%
        pred_set = np.zeros_like(true_set)

        # 计算在当前百分比阈值下应选择的正样本数量
        num_positive = max(1, int(percentile / 100 * len(sim_flat)))

        # 获取相似度最高的前 num_positive 个组合
        top_indices = np.argsort(-sim_flat)[:num_positive]

        # 将这些组合标记为正样本
        pred_set[top_indices] = 1

        # 计算指标
        precision = precision_score(true_set, pred_set)
        recall = recall_score(true_set, pred_set)
        f1 = f1_score(true_set, pred_set)

        # 将百分比形式的阈值存储
        percentile_str = f"{percentile / 100:.2f}"

        # 保存结果
        results.append({
            'Threshold': percentile_str,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
        # print(f'{percentile}%阈值下的相似度排名:{top_indices}')
        # print(f'选择的文档数量:{num_positive}')
        # print(f'选择的前百分比的：{top_indices}')
        # print(f'this is pred_set:{pred_set}')
    with pd.ExcelWriter(output_excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        results_df = pd.DataFrame(results)
        results_df.to_excel(writer, sheet_name=dataset_name, index=False)

