import psutil
import time

def print_process_info(proc):
    try:
        pid = proc.pid
        name = proc.name()
        cpu_percent = proc.cpu_percent(interval=0.1)
        memory_info = proc.memory_info()
        memory_percent = proc.memory_percent()
        
        print(f"PID: {pid}, Name: {name}, CPU: {cpu_percent}%, Memory: {memory_info.rss / 1024**2:.2f} MB ({memory_percent}%)")
        
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        pass

def find_processes_by_name(name):
    processes = []
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] == name:
                processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return processes

def get_child_processes_by_name(process_name):
    parent_processes = find_processes_by_name(process_name)
    if not parent_processes:
        print(f"No processes found with name: {process_name}")
        return

    for parent in parent_processes:
        print(f"Parent PID: {parent.pid}, Name: {parent.name()}")
        print_process_info(parent)
        
        children = parent.children(recursive=True)
        if children:
            print("Child processes:")
            for child in children:
                print_process_info(child)
        else:
            print("No child processes.")
        print("-" * 40)

# 示例：指定主进程的名字
main_process_name = "executor_runner" 
#main_process_name = "pt_main_thread"  # 将此处替换为实际的主进程名字

try:
    while True:
        time.sleep(1)
        get_child_processes_by_name(main_process_name)

except KeyboardInterrupt:
        print("program stop")
