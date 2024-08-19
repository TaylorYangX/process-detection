from executorch.extension.pybindings import portable_lib

import time

# Import torch after importing and using portable_lib to demonstrate that
# portable_lib works without importing this first.
import torch
m = portable_lib._load_for_executorch("a8w4_1_test_torchao.pte")


t = torch.randn((1,100,51))

start_time = time.time()
outputs = m.forward([t])
end_time = time.time()

elapsed_time = end_time - start_time
print(f"model run : {elapsed_time} seconds")
