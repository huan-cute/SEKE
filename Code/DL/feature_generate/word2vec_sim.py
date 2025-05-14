import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os


# 读取 Excel 文件中的 uc 和 cc 向量
def load_vectors_from_excel(file_path):
    uc_df = pd.read_excel(file_path, sheet_name='uc')
    cc_df = pd.read_excel(file_path, sheet_name='cc')
    uc_vectors = uc_df.iloc[:, 1:].values
    cc_vectors = cc_df.iloc[:, 1:].values
    uc_files = uc_df.iloc[:, 0].apply(lambda x: os.path.splitext(x)[0]).tolist()
    cc_files = cc_df.iloc[:, 0].tolist()

    return uc_vectors, cc_vectors, uc_files, cc_files


# 计算余弦相似度矩阵
def compute_similarity_matrix(uc_vectors, cc_vectors):
    return cosine_similarity(uc_vectors, cc_vectors)


# 根据排名选取前 x% 的链接
def select_top_percent_links(similarity_matrix, percentage):
    flat_similarities = similarity_matrix.flatten()
    sorted_indices = np.argsort(flat_similarities)[::-1]  # 从大到小排序
    total_links = len(flat_similarities)

    # 计算前 percentage 百分比的数量
    cutoff_index = int(np.ceil(total_links * percentage))  # 使用 np.ceil 来确保获得足够的链接
    top_indices = sorted_indices[:cutoff_index]
    selected_links = np.unravel_index(top_indices, similarity_matrix.shape)

    return list(zip(selected_links[0], selected_links[1]))


# 加载真集
def load_true_set(true_set_file_path):
    true_set = set()
    with open(true_set_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            uc_file = os.path.splitext(parts[0])[0]
            cc_files = parts[1:]
            for cc_file in cc_files:
                true_set.add((uc_file, cc_file))
    return true_set


# 计算评价指标
def evaluate(true_set, predicted_set, uc_file_list, cc_file_list):
    predicted_links = set((uc_file_list[i], cc_file_list[j]) for i, j in predicted_set)
    true_positives = len(predicted_links & true_set)
    predicted_positives = len(predicted_links)
    actual_positives = len(true_set)

    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    recall = true_positives / actual_positives if actual_positives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1


# 保存所有阈值下的评价指标
def save_metrics_to_excel(metrics, output_file_path):
    with pd.ExcelWriter(output_file_path) as writer:
        for dataset, dataset_metrics in metrics.items():
            dataset_metrics.to_excel(writer, sheet_name=dataset, index=False)


def main():
    # datasets = ['Infinispan']  # 数据集列表
    # 指定包含所有数据集的目录
    dataset_dir = '../dataset/uc/'
    # 获取所有数据集的名称（即每个子文件夹的名称）
    datasets = os.listdir(dataset_dir)
    base_path = '../dataset/'
    true_set_base_path = f'{base_path}/true_set'
    result_file_path = '../result/word2vec_sim_result/metrics.xlsx'
    percentages = np.arange(0.01, 1.01, 0.01)  # 百分比范围从 1% 到 100%

    metrics = {}

    for dataset in datasets:
        file_path = f'../result/word2vec_vec_result/{dataset}/{dataset}_word2vec_vectors.xlsx'
        true_set_file_path = f'{true_set_base_path}/{dataset}.txt'

        # 加载向量和真集
        uc_vectors, cc_vectors, uc_files, cc_files = load_vectors_from_excel(file_path)
        true_set = load_true_set(true_set_file_path)
        print(f'true_set size: {len(true_set)}')
        print(f'true_set:{true_set}')
        # 计算相似度矩阵
        similarity_matrix = compute_similarity_matrix(uc_vectors, cc_vectors)
        print(f'similarity_matrix:{similarity_matrix}')
        # 存储每个百分比下的评价指标
        dataset_metrics = []

        for percentage in percentages:
            # 根据当前百分比选取链接
            predicted_set = select_top_percent_links(similarity_matrix, percentage)
            if percentage == 1:
                print(f'predicted_set size: {len(predicted_set)}')
                print(f'predicted_set:{predicted_set}')
            # 计算评价指标
            precision, recall, f1 = evaluate(true_set, predicted_set, uc_files, cc_files)

            # 存储结果
            dataset_metrics.append({
                'Threshold': f'{percentage:.2f}',
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            })

        # 将结果保存到数据集对应的 sheet 中
        metrics[dataset] = pd.DataFrame(dataset_metrics)

    # 保存所有百分比下的评价指标到单一的 Excel 文件
    save_metrics_to_excel(metrics, result_file_path)

    print(f'All metrics for datasets have been saved to {result_file_path}')


if __name__ == '__main__':
    main()
