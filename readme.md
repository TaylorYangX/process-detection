after run the 3 kinds of quantized pte model and the orignal pth model
I find the process of running executorch with cpp is executor_runner(which is the executable file we create)
I find the process of running orignal python model and executorch with python  is pt_main_thread(which is the process of pytorch)


we can get result of cpu usage and memory usage.
| model          | process cpu usage | process memory usage | process memory percent | system cpu usage | system memory usage | system usage percent |
| -------------- | ----------------- | -------------------- | ---------------------- | ---------------- | ------------------- | -------------------- |
| a8w4 python    | 100.14%           | 341.63 MB            | 4.32%                  | 26.32%           | 3086.19 MB          | 44.00%               |
| a8w4  cpp      | 99.40%            | 45.46 MB             | 0.58%                  | 25.80%           | 2969.43 MB          | 42.21%               |
| a8w8 python    | 99.85%            | 337.18 MB            | 4.27%                  | 26.09%           | 3062.66 MB          | 43.73%               |
| a8w8 cpp       | 98.76%            | 36.93 MB             | 0.47%                  | 25.98%           | 2962.71 MB          | 42.12%               |
| w8only python  | 100.56%           | 333.69 MB            | 4.22%                  | 27.40%           | 3525.04 MB          | 49.53%               |
| w8only cpp     | 99.68%            | 36.55 MB             | 0.46%                  | 25.92%           | 2961.53 MB          | 42.19%               |
| orignal python | 297.69%           | 413.52 MB            | 5.23%                  | 99.18%           | 3165.48 MB          | 44.76%               |

the result with python on cpu

| model   | Accuracy | F-score  | Latency(s)         | Model Size(bytes) |
| ------- | -------- | -------- | ------------------ | ----------------- |
| a8w4    | 0.985513 | 0.942169 | 2.8647997991299134 | 20766072          |
| a8w8    | 0.982718 | 0.931772 | 2.6266351785839372 | 20332280          |
| w8only  | 0.980928 | 0.925058 | 2.46934628935588   | 20246248          |
| origanl | 0.9911   | 0.9646   | 4.689765601493677  | 29685120          |

![the radar chart](./Radar%20chart.png)

