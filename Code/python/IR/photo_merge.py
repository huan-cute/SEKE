from PIL import Image
import os


def merge_images(image_paths, output_path, rows, cols):
    """
    将多张图片拼接成一个大图片。

    :param image_paths: 图片文件路径列表
    :param output_path: 最终合成图片的输出路径
    :param rows: 行数
    :param cols: 列数
    """
    # 打开所有图片并计算最大宽度和高度
    images = [Image.open(img) for img in image_paths]

    # 获取每个图片的宽度和高度，假设所有图片大小相同
    widths, heights = zip(*(i.size for i in images))

    # 设置大图的尺寸
    total_width = max(widths) * cols
    total_height = max(heights) * rows

    # 创建一张新图片用于拼接
    new_image = Image.new('RGB', (total_width, total_height))

    # 把每张图片按顺序粘贴到大图里
    for index, img in enumerate(images):
        row = index // cols
        col = index % cols
        x_offset = col * max(widths)
        y_offset = row * max(heights)
        new_image.paste(img, (x_offset, y_offset))

    # 保存合成后的图片
    new_image.save(output_path)


# 示例使用
image_folder = '../IR/result_photo/PR'
# 获取所有图片的路径（确保文件夹中有15张图片）
image_files = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder)]

# 检查图片数量是否符合条件
if len(image_files) >= 15:
    # 前八张图片拼成一个2行4列的大图片
    output_file_1 = '../IR/result_photo/merged_image_1&gpt.png'
    merge_images(image_files[:8], output_file_1, rows=2, cols=4)

    # 后七张图片拼成一个2行4列的大图片（最后一个空白）
    output_file_2 = '../IR/result_photo/merged_image_2&gpt.png'
    merge_images(image_files[8:], output_file_2, rows=2, cols=4)
else:
    print("文件夹中的图片数量不足15张，请检查文件数量。")
