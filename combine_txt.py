
import os

for i in range(1,301):
    # 定义路径
    original_file_path = f'/home/yanwenli/light-weight-refinenet/test/data/{i}/agr_output_single.txt'
    new_data_file_path = f'/home/yanwenli/light-weight-refinenet/test/out_put_image/data_txt{i}.txt'
    output_file_path = f'/home/yanwenli/light-weight-refinenet/test/output_txt/combined_output{i}.txt'  # 确保这是一个文件路径，而不是目录路径

    # 检查文件是否存在
    if not os.path.exists(original_file_path):
        print(f"文件 {original_file_path} 不存在，跳过...")
        continue

    if not os.path.exists(new_data_file_path):
        print(f"文件 {new_data_file_path} 不存在，跳过...")
        continue

    # 读取原文件内容
    with open(original_file_path, 'r') as file:
        original_lines = file.readlines()

    # 读取新增数据文件中的一行数据
    with open(new_data_file_path, 'r') as file:
        new_data_line = file.readline().strip()

    # 合并数据：将新增数据文件中的这一行数据添加到原文件的每一行后面
    merged_lines = [original_line.strip() + ' ' + new_data_line + '\n' for original_line in original_lines]

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 写回合并后的数据到新的文件
    with open(output_file_path, 'w') as file:
        file.writelines(merged_lines)

    print(f"合并后的数据已保存到: {output_file_path}")


