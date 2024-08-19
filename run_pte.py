import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
#from utils.utils import *
#from Amodel.AnomalyTransformer import AnomalyTransformer
from SWaT_loader import get_loader_segment
from executorch.extension.pybindings import portable_lib
import time
from model.AnomalyTransformer import AnomalyTransformer

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def Origanltest():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_path = "/home/taylor/Project/testSWaT"
    dataset = "SWaT"
    batch_size =1
    win_size =100
    model = AnomalyTransformer(win_size=100, enc_in=51, c_out=51, e_layers=3)
    train_loader = get_loader_segment(data_path, batch_size=batch_size, win_size=win_size,
                                            mode='train',
                                            dataset=dataset)
    model.load_state_dict(
        torch.load(
            'ture256SWaT_checkpoint.pth',map_location=device, weights_only=True))
    model.eval()
    temperature = 50

    print("======================TEST MODE======================")

    criterion = nn.MSELoss(reduce=False)

    # (1) stastic on the train set
    attens_energy = []
    start_time = time.time()
    model_run_time = 0
    loop_num =0
    for i, (input_data, labels) in enumerate(train_loader):
        input = input_data.float().to(device)
        if input.shape != (batch_size,win_size,51):
            break
        model_start_time = time.time()
        output, series, prior, _ = model(input)
        model_end_time = time.time()
        model_run_time = model_run_time + model_end_time - model_start_time

        loss = torch.mean(criterion(input, output), dim=-1)
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)),
                    series[u].detach()) * temperature

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)
        loop_num = loop_num + 1

    end_time = time.time()
    # 计算运行时间
    elapsed_time = end_time - start_time
    print("loop number is ",loop_num)
    print(f"model run : {model_run_time/loop_num} seconds")
    print(f"model run and kl loss: {elapsed_time/loop_num} seconds")

def a8w4test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = portable_lib._load_for_executorch("a8w4_1_test_torchao.pte")
    data_path = "/home/taylor/Project/testSWaT"
    dataset = "SWaT"
    batch_size =1
    win_size =100
    train_loader = get_loader_segment(data_path, batch_size=batch_size, win_size=win_size,
                                            mode='train',
                                            dataset=dataset)
    
    temperature = 50

    print("======================TEST a8w4======================")

    criterion = nn.MSELoss(reduce=False)

    # (1) stastic on the train set
    attens_energy = []
    start_time = time.time()
    model_run_time = 0
    loop_num =0
    for i, (input_data, labels) in enumerate(train_loader):
        print("model run start")
        input = input_data.float().to(device)
        if input.shape != (batch_size,win_size,51):
            break
        model_start_time = time.time()
        outputs= model.forward([input])
        model_end_time = time.time()
        model_run_time = model_run_time + model_end_time - model_start_time
        output = outputs[0]
        series = {}
        prior={}
        for k in range(3):
            series[k] = outputs[k+1]
            prior[k] = outputs[k+4]
        loss = torch.mean(criterion(input, output), dim=-1)
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)),
                    series[u].detach()) * temperature

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)
        loop_num = loop_num + 1
    end_time = time.time()
    # 计算运行时间
    elapsed_time = end_time - start_time
    print("loop number is ",loop_num)
    print(f"model run : {model_run_time/loop_num} seconds")
    print(f"model run and kl loss: {elapsed_time/loop_num} seconds")

def a8w8test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = portable_lib._load_for_executorch("a8w8_1_test_torchao.pte")
    data_path = "/home/taylor/Project/testSWaT"
    dataset = "SWaT"
    batch_size =1
    win_size =100
    train_loader = get_loader_segment(data_path, batch_size=batch_size, win_size=win_size,
                                            mode='train',
                                            dataset=dataset)
    
    temperature = 50

    print("======================TEST a8w8======================")

    criterion = nn.MSELoss(reduce=False)

    # (1) stastic on the train set
    attens_energy = []
    start_time = time.time()
    model_run_time = 0
    loop_num =0
    for i, (input_data, labels) in enumerate(train_loader): 
        input = input_data.float().to(device)
        if input.shape != (batch_size,win_size,51):
            break
        model_start_time = time.time()
        outputs= model.forward([input])
        model_end_time = time.time()
        model_run_time = model_run_time + model_end_time - model_start_time
        output = outputs[0]
        series = {}
        prior={}
        for k in range(3):
            series[k] = outputs[k+1]
            prior[k] = outputs[k+4]
        loss = torch.mean(criterion(input, output), dim=-1)
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)),
                    series[u].detach()) * temperature

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)
        loop_num = loop_num + 1
    end_time = time.time()
    # 计算运行时间
    elapsed_time = end_time - start_time
    print("loop number is ",loop_num)
    print(f"model run : {model_run_time/loop_num} seconds")
    print(f"model run and kl loss: {elapsed_time/loop_num} seconds")

def w8onlytest():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = portable_lib._load_for_executorch("w8only_1_test_torchao.pte")
    data_path = "/home/taylor/Project/testSWaT"
    dataset = "SWaT"
    batch_size =1
    win_size =100
    train_loader = get_loader_segment(data_path, batch_size=batch_size, win_size=win_size,
                                            mode='train',
                                            dataset=dataset)
    
    temperature = 50

    print("======================TEST weight int 8 only ======================")

    criterion = nn.MSELoss(reduce=False)

    # (1) stastic on the train set
    attens_energy = []
    start_time = time.time()
    model_run_time = 0
    loop_num =0
    for i, (input_data, labels) in enumerate(train_loader):
        input = input_data.float().to(device)
        if input.shape != (batch_size,win_size,51):
            break
        model_start_time = time.time()
        outputs= model.forward([input])
        model_end_time = time.time()
        model_run_time = model_run_time + model_end_time - model_start_time
        output = outputs[0]
        series = {}
        prior={}
        for k in range(3):
            series[k] = outputs[k+1]
            prior[k] = outputs[k+4]
        loss = torch.mean(criterion(input, output), dim=-1)
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                win_size)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)),
                    series[u].detach()) * temperature

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)
        loop_num = loop_num + 1
    end_time = time.time()
    # 计算运行时间
    elapsed_time = end_time - start_time
    print("loop number is ",loop_num)
    print(f"model run : {model_run_time/loop_num} seconds")
    print(f"model run and kl loss: {elapsed_time/loop_num} seconds")

if __name__== "__main__" :
    #a8w4test()
    #a8w8test()
    w8onlytest()
    #Origanltest()
