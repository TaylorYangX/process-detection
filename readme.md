after run the 3 kinds of quantized pte model and the orignal pth model
I find the process of running executorch with cpp is executor_runner(which is the executable file we create)
I find the process of running orignal python model and executorch with python  is pt_main_thread(which is the process of pytorch)


we can get result of cpu usage and memory usage.


| model          | cpu usage | memory usage | memory percent |
| -------------- | --------- | ------------ | -------------- |
| a8w4 python    | 100.14%   | 341.63 MB    | 4.32%          |
| a8w4  cpp      | 99.40%    | 45.46 MB     | 0.58%          |
| a8w8 python    | 99.85%    | 337.18 MB    | 4.27%          |
| a8w8 cpp       | 98.76%    | 36.93 MB     | 0.47%          |
| w8only python  | 100.56%   | 333.69 MB    | 4.22%          |
| w8only cpp     | 99.68%    | 36.55 MB     | 0.46%          |
| orignal python | 297.69%   | 413.52 MB    | 5.23%          |

