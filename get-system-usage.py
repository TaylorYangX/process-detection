import psutil
import time

def system_usage():
    # 获取CPU使用率，interval表示计算的时间间隔
    cpu_usage = psutil.cpu_percent()

    # 获取内存使用情况
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent

    print(f"CPU: {cpu_usage}%, Memory: {memory_info.used / 1024**2:.2f} MB ({memory_usage}%)")

try:
    while True:
        time.sleep(1)
        system_usage()

except KeyboardInterrupt:
        print("program stop")
