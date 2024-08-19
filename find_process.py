#使用一个无限循环来每两秒比较一次进程列表，并找出新增的进程。
import psutil
import time
import csv


def get_process_list():
    return {p.pid: p.info for p in psutil.process_iter(['pid', 'name'])}

# get the initial process list
initial_processes = get_process_list()

filename = 'process_list.csv'

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # 写入CSV头
    writer.writerow(['PID', 'Name'])

    try:
        while True:
            # detect the process every 2 s
            time.sleep(1)
            
            # get the new process list
            new_processes = get_process_list()
            
            # find the new process
            added_processes = {pid: info for pid, info in new_processes.items() if pid not in initial_processes}
            
            if added_processes:
                print("the new process:")
                writer.writerow([' ',' '])
                for pid, info in added_processes.items():
                    print(f"PID: {pid}, Name: {info['name']}")
                    # 写入进程信息到CSV文件
                    writer.writerow([pid, info['name']])
            
            # updata the process list
            #initial_processes = new_processes

    except KeyboardInterrupt:
        print("program stop")