import pandas as pd
import matplotlib.pyplot as plt
import os

def calculate_map(precision, recall):
    """
    计算MAP值，根据精确率和召回率。
    MAP值是所有召回率点上的平均精确率。
    :param precision: 精确率列表
    :param recall: 召回率列表
    :return: 平均精确率 (AP)
    """
    # 初始化平均精确率 (AP)
    ap = 0.0
    relevant_documents = 0

    for i in range(len(recall)):
        if recall[i] > 0:  # 只计算相关文档处的精确率
            ap += precision[i]
            relevant_documents += 1

    # 返回MAP值（如果没有相关文档，返回0）
    return ap / relevant_documents if relevant_documents > 0 else 0

def plot_map_bar_chart(excel_files, output_dir):
    """
    读取多个方法的Excel文件，提取相同数据集名称的sheet，计算MAP值并绘制柱状图。

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
        map_values = {}

        # 对每个方法的相同sheet进行处理，计算MAP值
        for method, xls in all_sheets.items():
            df = pd.read_excel(xls, sheet_name=sheet_name)

            # 假设Excel文件中包含'Threshold', 'Precision', 'Recall'列
            precision = df['Precision']
            recall = df['Recall']

            # 计算MAP值
            map_value = calculate_map(precision, recall)
            map_values[method] = map_value

        # 绘制MAP值柱状图
        plt.figure(figsize=(10, 6))
        plt.bar(map_values.keys(), map_values.values(), color='#ADD8E6')  # 浅蓝色

        # 设置图表标题和标签
        plt.title(f'{sheet_name} - MAP Comparison')
        plt.xlabel('Method')
        plt.ylabel('MAP')

        # 保存图像
        output_file = os.path.join(output_dir, f'{sheet_name}_map_comparison.png')
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
output_dir = '../IR/result_photo/MAP'
plot_map_bar_chart(excel_files, output_dir)
