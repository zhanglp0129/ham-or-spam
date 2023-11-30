import torch

batch_size = 128
epochs = 50
learning_rate = 0.001
hidden_size = 256
num_layers = 2
embedding_size = 300
dropout = 0.5

# 定义：ham为0，spam为1
ham = 0
spam = 1

mean = 645.2935959181727
std = 884.3463761591171

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型保存路径
model_save_path = "./model/hamspam.pth"
# 模型导入路径
model_load_path = "./model/hamspam.pth"

# 数据集路径
data_path = "./data/spamdata_T2.csv"