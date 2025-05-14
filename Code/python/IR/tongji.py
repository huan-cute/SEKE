import pandas as pd
import os


def calculate_f1_mean(input_paths, output_path):
    # 用来存储结果的字典，key为算法名，value为对应算法的F1均值列表
    result_data = {}

    # 遍历输入路径列表
    for input_path in input_paths:
        if os.path.isfile(input_path):  # 如果是文件路径
            file_paths = [input_path]  # 直接处理这个文件
        else:
            print(f"Warning: {input_path} is not a valid file path.")
            continue  # 如果路径无效，则跳过此路径

        # 读取并处理每个文件
        for file_path in file_paths:
            try:
                # 提取文件名作为算法名
                algorithm_name = os.path.basename(file_path).split('.')[0]  # 取文件名作为算法名（去掉扩展名）

                # 初始化算法名对应的结果列表
                if algorithm_name not in result_data:
                    result_data[algorithm_name] = []

                # 读取Excel文件
                xls = pd.ExcelFile(file_path)
                for sheet_name in xls.sheet_names:
                    # 读取每个sheet的内容
                    df = xls.parse(sheet_name)

                    # 输出每个sheet的列名，帮助调试
                    print(f"Checking file: {file_path}, sheet: {sheet_name}")
                    print("Columns:", df.columns)

                    # 确保F1值所在的列是已知的，假设列名为'F1 Score'或'F1'
                    f1_mean = None
                    if 'F1 Score' in df.columns:
                        f1_mean = df['F1 Score'].mean()
                    elif 'F1' in df.columns:
                        f1_mean = df['F1'].mean()

                    if f1_mean is not None:
                        # 将算法名和对应sheet的F1均值加入结果字典
                        result_data[algorithm_name].append({'sheet': sheet_name, 'F1_mean': f1_mean})
                    else:
                        print(f"Warning: No F1 column found in sheet {sheet_name} of file {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    # 如果没有数据，打印警告
    if not result_data:
        print("Warning: No F1 scores found in the given files.")

    # 将结果保存到Excel，每个算法一个sheet
    if result_data:
        with pd.ExcelWriter(output_path) as writer:
            for algorithm_name, f1_results in result_data.items():
                # 将每个算法的F1均值列表转换为DataFrame
                result_df = pd.DataFrame(f1_results)
                # 将DataFrame写入对应的sheet
                result_df.to_excel(writer, sheet_name=algorithm_name, index=False)

        print(f"Results have been saved to {output_path}")
    else:
        print("No results to save.")


# 示例用法
input_paths = [
    'D:\desktop\\2024Paper\Code\pythonProject2\IR\BM25_results.xlsx',
    'D:\desktop\\2024Paper\Code\pythonProject2\IR\JS_results.xlsx',
    'D:\desktop\\2024Paper\Code\pythonProject2\IR\LDA_results.xlsx',
    'D:\desktop\\2024Paper\Code\pythonProject2\IR\LSI_results.xlsx',
    'D:\desktop\\2024Paper\Code\pythonProject2\IR\VSM_results.xlsx'
]  # 替换为实际文件路径
output_path = 'D:/desktop/2024Paper/Code/pythonProject2/Res/IR_f1_means_method.xlsx'  # 替换为输出文件路径

calculate_f1_mean(input_paths, output_path)
