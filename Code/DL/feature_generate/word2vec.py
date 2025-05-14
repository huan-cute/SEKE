import gensim
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# 读取文本文件
def read_texts(file_path):
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        texts = file.readlines()
    texts = [text.strip() for text in texts]
    return texts

# 加载预训练的 Word2Vec 模型
def load_pretrained_word2vec_model(model_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
    return model

# 获取每个文本的向量
def get_text_vector(model, text):
    words = text.split()
    word_vectors = [model[word] for word in words if word in model]
    if word_vectors:
        vector = np.mean(word_vectors, axis=0)
    else:
        vector = np.zeros(model.vector_size)
    return vector

# 将向量写入 Excel 文件的不同 sheet，并包括文件名
def write_vectors_to_excel(vectors_uc, vectors_cc, uc_files, cc_files, file_path):
    with pd.ExcelWriter(file_path) as writer:
        if vectors_uc:
            df_uc = pd.DataFrame(vectors_uc)
            df_uc.insert(0, 'File Name', uc_files)  # 将文件名作为第一列
            df_uc.to_excel(writer, sheet_name='uc', index=False)

        if vectors_cc:
            df_cc = pd.DataFrame(vectors_cc)
            df_cc.insert(0, 'File Name', cc_files)  # 将文件名作为第一列
            df_cc.to_excel(writer, sheet_name='cc', index=False)

# 主函数
def main():
    # 指定包含所有数据集的目录
    dataset_dir = '../dataset/uc/'
    # 获取所有数据集的名称（即每个子文件夹的名称）
    datasets = os.listdir(dataset_dir)
    types = ['uc', 'cc']  # 数据类型

    pretrained_model_path = '../GoogleNews-vectors-negative300.bin'

    # 检查预训练模型是否存在
    if not os.path.exists(pretrained_model_path):
        print(f"Pretrained model not found: {pretrained_model_path}")
        return

    # 加载预训练的 Word2Vec 模型
    model = load_pretrained_word2vec_model(pretrained_model_path)

    for dataset in datasets:
        vectors_uc, vectors_cc = [], []
        uc_files, cc_files = [], []  # 文件名列表

        for type in types:
            # 生成目录路径
            input_dir_path = f'../dataset/{type}/{dataset}/'

            # 检查目录是否存在
            if not os.path.exists(input_dir_path):
                print(f"Input directory not found: {input_dir_path}")
                continue

            # 遍历目录中的所有文件
            for file_name in tqdm(os.listdir(input_dir_path), desc=f"Processing {type} files in {dataset}"):
                # 生成文件路径
                input_file_path = os.path.join(input_dir_path, file_name)

                # 读取和预处理文本
                texts = read_texts(input_file_path)

                # 获取每个文件的向量
                file_vector = get_text_vector(model, ' '.join(texts))

                # 将向量根据类型存储，并记录文件名
                if type == 'uc':
                    vectors_uc.append(file_vector)
                    uc_files.append(file_name)  # 存储文件名
                elif type == 'cc':
                    vectors_cc.append(file_vector)
                    cc_files.append(file_name)  # 存储文件名

        # 输出文件路径
        output_file_path = f'../result/word2vec_vec_result/vec_result/{dataset}/{dataset}_word2vec_vectors.xlsx'

        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # 将向量与文件名写入 Excel 文件
        write_vectors_to_excel(vectors_uc, vectors_cc, uc_files, cc_files, output_file_path)
        print(f'Word2Vec vectors for dataset {dataset} have been written to {output_file_path}')

if __name__ == '__main__':
    main()
