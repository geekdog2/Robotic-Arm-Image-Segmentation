import os

# 定义路径和范围
input_folder = '/home/yanwenli/light-weight-refinenet/test/output_txt/'
output_file_path = '/home/yanwenli/light-weight-refinenet/test/output_txt/final_combined_output.txt'

# 输入文件的序号范围
file_range = range(1, 301)

# 存储所有输入文件内容
all_lines = []

# 读取所有输入文件内容并添加到all_lines
for i in file_range:
    file_path = f'{input_folder}combined_output{i}.txt'
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            all_lines.extend(lines)  # 添加所有行到all_lines
    else:
        print(f"文件 {file_path} 不存在，跳过...")

# 确保输出目录存在
output_dir = os.path.dirname(output_file_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 写入合并后的数据到新的文件
with open(output_file_path, 'w') as output_file:
    output_file.writelines(all_lines)

print(f"所有文件合并后的数据已保存到: {output_file_path}")
