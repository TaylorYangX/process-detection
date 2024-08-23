import numpy as np
import matplotlib.pyplot as plt
from math import pi

# 数据准备
labels = ['Mean Process-wise CPU Usage','Mean Process-wise Memory Usage','Mean Process-wise Memory Percent','Mean System-wide CPU Usage',
          'Mean System-wide Memory Usage','Mean System-wide Memory Percent','Accuracy', 'F-score', 'Latency', 'Model Size'
          ]

num_vars = len(labels)

# 样本数据
base_line = [100 , 200 , 40 , 100 , 250 , 40 , 1 , 1 , 7 , 30000000]
expand_difference = [0 , 300 , 0 , 0 ,2000 , 0 , 0 , 0 , 0 ,0]

a8w4_python  = [25.20 , 343.83 , 4.35 , 26.32 , 2166.25 , 31.97 , 0.985513 , 0.942169 , 2.8647997991299134 , 20766072]
a8w4_python_result = [(a-c) / b for a, b,c in zip(a8w4_python, base_line,expand_difference)]

a8w8_python = [25.24 , 335.55 , 4.25 ,  26.18 ,  2047.07 , 30.43 ,0.982718 , 0.931772 , 2.6266351785839372 , 20332280]
a8w8_python_result = [(a-c) / b for a, b,c in zip(a8w8_python, base_line,expand_difference)]

w8only_python = [25.23 , 336.43 ,  4.26 ,  25.75 ,  2174.38 , 32.08 , 0.980928 , 0.925058 , 2.46934628935588 , 20246248]
w8only_python_result = [(a-c) / b for a, b,c in zip(w8only_python, base_line,expand_difference)]

orignal_python = [85.89 , 445.73 ,  5.64 ,  98.84 ,  2206.37 , 32.23 , 0.9911,  0.9646 , 4.689765601493677 , 29685120]
orignal_python_result = [(a-c) / b for a, b,c in zip(orignal_python, base_line,expand_difference)]

#result = [a / b for a, b in zip(list1, list2)] use this code to divide one by one

values = [a8w4_python_result,a8w8_python_result,w8only_python_result,orignal_python_result]

colors = ['red', 'green',  'darkblue', 'cyan']

# 设置子图
fig, axes = plt.subplots(2, 2, subplot_kw=dict(polar=True), figsize=(12, 12))
axes = axes.flatten()


# 绘制每个雷达图
for i, ax in enumerate(axes):
    # 计算每个轴的角度
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    # 将第一个值添加到末尾以闭合圆形
    data = values[i] + values[i][:1]

    # 绘制和填充
    ax.plot(angles, data, color=colors[i], linewidth=2, linestyle='solid')
    ax.fill(angles, data, color=colors[i], alpha=0.4)

    # 设置标签和标题
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(range(1, 11))
    ax.set_yticks([0.1, 0.3, 0.7])  # 设置刻度位置
    ax.set_yticklabels(['0.1', '0.3', '0.7'], color="grey", size=7)


# 添加文字说明
description = (
    '     \n' '     \n' '     \n' '1.Mean Process-wise CPU Usage    ' '2.Mean Process-wise Memory Usage    ' '3.Mean Process-wise Memory Percent\n' '4.Mean System-wide CPU Usage    '
          '5.Mean System-wide Memory Usage    ' '6.Mean System-wide Memory Percent\n' '7.Accuracy    ' '8.F-score    ' '9.Latency    '  '10.Model Size'
)

# 设置文字位置 (图的底部)
plt.figtext(0.5, 0.02, description, ha="center", fontsize=12, wrap=True)

# 设置图例
legend_labels = [
    'A8W4 Dynamic Quantization',
    'A8W8 Dynamic Quantization',
    'A16W8 WeightOnly Quantization',
    'Float 32 bit (Non Quantized)'
]

fig.legend(legend_labels, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize='large')

# 保存图像
plt.savefig('Radar_chart.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
