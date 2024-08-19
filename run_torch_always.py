import torch
import time




try:
    while True:
        tensor1 =  torch.randn(2,2)
        tensor2 =  torch.randn(2,2)
        print("the sum of tensor 1 and 2 is",tensor1 + tensor2)

except KeyboardInterrupt:
    print("program stop")