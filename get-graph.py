import numpy as np
import matplotlib.pyplot as plt
from math import pi

# 数据准备
labels = ['Mean Process-wise CPU Usage','Mean Process-wise Memory Usage','Mean Process-wise Memory Percent','Mean System-wide CPU Usage',
          'Mean System-wide Memory Usage','Mean System-wide Memory Percent','Accuracy', 'F-score', 'Latency', 'Model Size'
          ]

num_vars = len(labels)

# 样本数据
base_line = [400 , 3500 , 100 , 400 , 3500 , 100 , 1 , 1 , 10 , 30000000]

a8w4_python  = [100.14  , 341.63 ,  4.32 , 26.32 , 3086.19  , 44.00 , 0.985513 , 0.942169 , 2.8647997991299134 , 20766072]
a8w4_python_result = [a / b for a, b in zip(a8w4_python, base_line)]

a8w8_python = [99.85 , 337.18  , 4.27 , 26.09 , 3062.66  , 43.73,0.982718 , 0.931772 , 2.6266351785839372 , 20332280]
a8w8_python_result = [a / b for a, b in zip(a8w8_python, base_line)]

w8only_python = [100.56 , 333.69  , 4.22 , 27.40 , 3525.04  , 49.53,0.980928 , 0.925058 , 2.46934628935588 , 20246248]
w8only_python_result = [a / b for a, b in zip(w8only_python, base_line)]

orignal_python = [297.69 , 413.52  , 5.23 , 99.18 , 3165.48  , 44.76,0.9911,  0.9646 , 4.689765601493677 , 29685120]
orignal_python_result = [a / b for a, b in zip(orignal_python, base_line)]

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
    ax.set_yticklabels(['0', '0.5', '1'], color="grey", size=7)


# 添加文字说明
description = (
    '     \n' '     \n' '     \n' 'Mean Process-wise CPU Usage' 'Mean Process-wise Memory Usage' 'Mean Process-wise Memory Percent\n' 'Mean System-wide CPU Usage'
          'Mean System-wide Memory Usage' 'Mean System-wide Memory Percent\n' 'Accuracy' 'F-score' 'Latency'  'Model Size'
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
plt.savefig('Radar chart.png', dpi=300, bbox_inches='tight')

# 显示图像
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
