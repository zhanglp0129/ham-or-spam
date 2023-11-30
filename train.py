import torch
import numpy as np
from tqdm import tqdm
from torch import nn, optim
import matplotlib.pyplot as plt
import conf
import data
import models

def train(model:nn.Module, optimizer:optim.Optimizer, criterion):
    loss_list = []
    acc_list = []
    for i in range(conf.epochs):
        # 训练模型
        model.train()
        dataloader = data.getDataloader(True)
        for inputs, labels in dataloader:
            inputs = inputs.to(conf.device)
            labels = labels.to(conf.device)
            optimizer.zero_grad()
            predicts = model(inputs)
            loss = criterion(predicts, labels)
            loss.backward()
            optimizer.step()

        # 评估模型
        eval_loss, eval_acc = eval(model, criterion)
        loss_list.append(eval_loss)
        acc_list.append(eval_acc)
        print(f"{i+1}/{conf.epochs}，损失：{eval_loss}，正确率：{eval_acc}")
        torch.save(model.state_dict(), conf.model_save_path)
    return loss_list, acc_list


def eval(model:nn.Module, criterion)->(float, float):
    model.eval()
    loss_list = []
    acc_list = []

    dataloader = data.getDataloader(False)
    for inputs, labels in dataloader:
        with torch.no_grad():
            inputs = inputs.to(conf.device)
            labels = labels.to(conf.device)
            predicts = model(inputs)
            loss = criterion(predicts, labels).item()
            loss_list.append(loss)

            pred = predicts.squeeze(1) > 0.5
            label = labels.squeeze(1) > 0.5
            acc_list.append(torch.sum(pred == label).item() / pred.size(0))
    return np.mean(loss_list), np.mean(acc_list)



if __name__ == "__main__":
    model = models.HamSpamModel()
    model = model.to(conf.device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    loss, acc = train(model, optimizer, criterion)
    print(loss)
    print(acc)