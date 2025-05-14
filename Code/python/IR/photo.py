import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_precision_recall_comparison(excel_files, output_dir):
    """
    读取多个方法的Excel文件，提取相同数据集名称的sheet，绘制精确率和召回率对比图。

    :param excel_files: 包含所有方法的Excel文件路径字典，例如{'VSM': 'path/to/vsm.xlsx', 'BM25': 'path/to/bm25.xlsx', 'LSI': 'path/to/lsi.xlsx'}
    :param output_dir: 输出图像的目录
    """
    # 获取所有方法的Excel文件，并提取每个文件中的sheet名称
    sheet_names_set = None
    all_sheets = {}

    for method, excel_file in excel_files.items():
        xls = pd.ExcelFile(excel_file)
        all_sheets[method] = xls
        # 初始化sheet名称集合，确保所有方法中都有相同的数据集
        if sheet_names_set is None:
            sheet_names_set = set(xls.sheet_names)
        else:
            sheet_names_set &= set(xls.sheet_names)  # 只保留相同的sheet名称

    # 遍历所有相同名称的sheet（即相同数据集）
    for sheet_name in sheet_names_set:
        plt.figure(figsize=(10, 6))

        # 对每个方法的相同sheet进行处理
        for method, xls in all_sheets.items():
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # 假设Excel文件中包含'Threshold', 'Precision', 'Recall'列
            thresholds = df['Threshold']  # 如果不需要，可以忽略此行
            precision = df['Precision']
            recall = df['Recall']

            # 绘制精确率 vs 召回率 (Recall为横坐标, Precision为纵坐标)
            plt.plot(recall, precision, label=f'{method}', marker='o')

        # 设置图表标题和标签
        plt.title(f'{sheet_name} - Precision vs Recall Comparison')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()

        # 保存图像
        output_file = os.path.join(output_dir, f'{sheet_name}_comparison.png')
        plt.savefig(output_file)
        plt.close()

# 示例使用
excel_files = {
    'BM25': '../IR/BM25_results.xlsx',
    'JS': '../IR/JS_results.xlsx',
    'LDA': '../IR/LDA_results.xlsx',
    'LSI': '../IR/LSI_results.xlsx',
    'VSM': '../IR/VSM_results.xlsx'
}
output_dir = '../IR/result_photo/PR'

plot_precision_recall_comparison(excel_files, output_dir)
