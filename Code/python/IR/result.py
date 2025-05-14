import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from pre_generate import set_generation
from gensim import corpora, models, similarities
from xlsxwriter import Workbook
from process_dataset import process_dataset


if __name__ == '__main__':
    output_excel_path = 'LSI_Drools_results_final.xlsx'
    # 检查文件是否存在
    if not os.path.exists(output_excel_path):
        # 创建一个有效的 Excel 文件
        with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
            # 创建一个空的 DataFrame 来初始化文件
            df = pd.DataFrame()
            df.to_excel(writer)

    # 指定包含所有数据集的目录
    # dataset_dir = '../dataset/uc/'
    # # 获取所有数据集的名称（即每个子文件夹的名称）
    # dataset_names = os.listdir(dataset_dir)
    #遍历每个数据集进行处理
    dataset_names = ['Drools']
    for dataset_name in dataset_names:
        # if dataset_name == 'Drools':
        #     continue
        process_dataset(dataset_name,output_excel_path)
        print(f"{dataset_name} processed and saved to Excel.")
    print("All datasets processed and saved to Excel.")

