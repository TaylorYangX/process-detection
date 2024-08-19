import re

def extract_values_from_file(file_path):
    cpu_values = []
    memory_values = []
    memory_percentage_values = []

    # 使用正则表达式匹配 CPU、Memory 和括号内的百分率
    cpu_pattern = re.compile(r'CPU:\s*([\d.]+)%')
    memory_pattern = re.compile(r'Memory:\s*([\d.]+)\s*MB')
    memory_percentage_pattern = re.compile(r'\(([\d.]+)%\)')

    with open(file_path, 'r') as file:
        for line in file:
            # 查找 CPU、Memory 和括号内百分率的值
            cpu_match = cpu_pattern.search(line)
            memory_match = memory_pattern.search(line)
            memory_percentage_match = memory_percentage_pattern.search(line)

            if cpu_match and memory_match and memory_percentage_match:
                # 提取并转换为浮点数
                cpu_values.append(float(cpu_match.group(1)))
                memory_values.append(float(memory_match.group(1)))
                memory_percentage_values.append(float(memory_percentage_match.group(1)))

    if cpu_values and memory_values and memory_percentage_values:
        avg_cpu = sum(cpu_values) / len(cpu_values)
        avg_memory = sum(memory_values) / len(memory_values)
        avg_memory_percentage = sum(memory_percentage_values) / len(memory_percentage_values)
        return avg_cpu, avg_memory, avg_memory_percentage
    else:
        return None, None, None

# 替换为你的文件路径
file_path = '/home/taylor/Project/Ea8w4/a8w4-pt_main_thread.txt'
average_cpu, average_memory, average_memory_percentage = extract_values_from_file(file_path)

if average_cpu is not None and average_memory is not None and average_memory_percentage is not None:
    print(f"Average CPU: {average_cpu:.2f}%")
    print(f"Average Memory: {average_memory:.2f} MB")
    print(f"Average Memory Percentage: {average_memory_percentage:.2f}%")
else:
    print("No CPU, Memory or Percentage values found in the file.")
