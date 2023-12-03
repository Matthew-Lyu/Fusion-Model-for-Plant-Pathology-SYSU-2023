import argparse
import math
import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from loss_function import *
from visualize.ConfusionMatrix import *
import subprocess
import sys

from models import *
from dataset import LeafDataset

# 命令行传参
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--num_of_epoch', type = int, default=50, help='num of epoch')
parser.add_argument('--batch_size', type = int, default=8, help='batch size')
parser.add_argument('--lr', type = float, default=0.0001, help='learning rate')
parser.add_argument('--train_label_dir', default='/home/hbenke/Project/Lvwc/Project/cv/leaf_disease_classifier/dataset/train.csv', help='train label dir')
parser.add_argument('--train_image_dir', default='/home/hbenke/Project/Lvwc/Project/Data/leaf_disease/train/images/', help='train image dir')
parser.add_argument('--val_label_dir', default='/home/hbenke/Project/Lvwc/Project/cv/leaf_disease_classifier/dataset/val.csv', help='val label dir')
parser.add_argument('--val_image_dir', default='/home/hbenke/Project/Lvwc/Project/Data/leaf_disease/val/images/', help='val image dir')
parser.add_argument('--save_dir', default='/home/hbenke/Project/Lvwc/Project/cv/leaf_disease_classifier/runs/train_result/exp1.1/', help='save dir')
parser.add_argument('--weights_dir', default='', help='weights path')
args = parser.parse_args()

# 超参数
# 设置为gpu训练
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

num_of_epoch = args.num_of_epoch
learning_rate = args.lr
batch_size = args.batch_size

# 数据路径
train_label_dir = args.train_label_dir
train_image_dir = args.train_image_dir
val_label_dir = args.val_label_dir
val_image_dir = args.val_image_dir
save_dir = args.save_dir
weights_dir = args.weights_dir

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 获取数据集和dataloader
train_ds = LeafDataset(csv_file=train_label_dir, imgs_path=train_image_dir,
                transform=torchvision.transforms.Compose([
                    # [312, 1000]
                    torchvision.transforms.Resize([224, 224]),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
                    ]))

val_ds = LeafDataset(csv_file=val_label_dir, imgs_path=val_image_dir,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize([224, 224]),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
                    ]))

train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=4) # num_workers=4 表示用四个子进程加载数据
val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=True, num_workers=4)

TRAIN_SIZE = len(train_ds)
VALID_SIZE = len(val_ds)

# 二分类交叉熵损失
loss_fn = torch.nn.BCEWithLogitsLoss()

print_running_loss = False

# 训练函数
def train_fn(net, loader):

    tr_loss = 0
    tr_accuracy = 0
    
    # 创建一个迭代对象
    items = enumerate(loader)
    total_items = len(loader)  # 获取迭代对象的总长度
    
    for _, (images, labels) in tqdm(items, total=total_items, desc="train"):
        images, labels = images.to(device), labels.to(device)
        
        # 梯度置0
        optimizer.zero_grad()

        # 向前传播
        # predictions = net(images, labels)
        predictions = net(images)
        loss = loss_fn(predictions, labels.squeeze(-1))

        # 清理梯度
        net.zero_grad()

        # 反向传播
        loss.backward()
        tr_loss += loss.item()

        # 计算准确率

        ## 二分类交叉熵
        batch_shape = list(predictions.size())
        for i in range(batch_shape[0]):
            for j in range(batch_shape[1]):
                prediction = 1 if predictions.detach().cpu().numpy()[i][j] >= 0.5 else 0
                if prediction == labels.detach().cpu().numpy()[i][j]:
                    tr_accuracy += 1.0/batch_shape[1]


        optimizer.step()

        if print_running_loss and _ % 10 == 0:
            print("One image finished, running loss is" + str(tr_loss/TRAIN_SIZE))

    return tr_accuracy/TRAIN_SIZE, tr_loss/TRAIN_SIZE

# 验证函数
def valid_fn(net, loader):
    
    valid_loss = 0
    valid_accuracy = 0
    
    with torch.no_grad():       
        
        # 创建一个迭代对象
        items = enumerate(loader)
        total_items = len(loader)  # 获取迭代对象的总长度
        
        for _, (images, labels) in tqdm(items, total=total_items, desc="val"):
            
            images, labels = images.to(device), labels.to(device)
            net.eval()
            
            # predictions = net(images,labels)
            predictions = net(images)
            loss = loss_fn(predictions, labels.squeeze(-1))
            
            valid_loss += loss.item()

            ## 二分类交叉熵
            batch_shape = list(predictions.size())
            for i in range(batch_shape[0]):
                for j in range(batch_shape[1]):
                    prediction = 1 if predictions.detach().cpu().numpy()[i][j] >= 0.5 else 0
                    if prediction == labels.detach().cpu().numpy()[i][j]:
                        valid_accuracy += 1.0/batch_shape[1]
                 
    return valid_accuracy/VALID_SIZE, valid_loss/VALID_SIZE


# 训练
# 定义模型

leaf_model = ResNet50(num_classes=6, pretrained=True).to(device)

# leaf_model = DenseNet(num_classes=6).to(device)

# leaf_model = SwinTransformer(

#     hidden_dim=96,
#     layers=(2, 2, 6, 2),
#     heads=(3, 6, 12, 24),
#     channels=3,
#     num_classes=6,
#     head_dim=32,
#     window_size=7,
#     downscaling_factors=(4, 2, 2, 2),
#     relative_pos_embedding=True
    
# ).to(device)

# leaf_model = EfficientNet(num_classes=6).to(device)

# leaf_model = MobileNetV2(num_classes=6).to(device)

# leaf_model = VisionTransformer(
#     # image_size=224, 
#     # patch_size=16, 
#     # dim=768, 
#     # depth=12, 
#     # heads=8, 
#     # mlp_dim=3072,
#     num_classes=6
# ).to(device)

# leaf_model = AwesomeModel4(num_classes=6).to(device)

# leaf_model = AwesomeModel2(num_classes=6).to(device)

# leaf_model = AwesomeModel3(num_classes=6).to(device)

# leaf_model = AwesomeModel4(num_classes=6).to(device)


optimizer = optim.Adam(leaf_model.parameters(), lr=learning_rate)

train_loss = []
train_acc = []
valid_loss = []
valid_acc = []
train_acc = []
val_acc = []

log_file_path = os.path.join(save_dir, "train.log")

if __name__ == "__main__":

    best_val_acc = 0.
    val_predictions = []

    log_file = open(log_file_path, "w")

    # 训练
    for epoch in range(num_of_epoch):

        os.system(f'echo \"Epoch {epoch+1}\"')
    
        leaf_model.train()
        
        ta, tl = train_fn(leaf_model, loader=train_loader)
        va, vl = valid_fn(leaf_model, loader=val_loader)
        train_loss.append(tl)
        valid_loss.append(vl)
        train_acc.append(ta)
        valid_acc.append(va)

        log_file.write(f"Epoch {epoch + 1}:\tTrain Acc {ta:.4f}\tTrain Loss {tl:.4f}\tVal Acc {va:.4f}\tVal Loss {vl:.4f}\n")

        print('Epoch: '+ str(epoch) + ', Train loss: ' + str(tl) + ', Train accuracy: ' + str(ta)
            + ', Val loss: ' + str(vl) + ', Val accuracy: ' + str(va))

        if va > best_val_acc:
             best_val_acc = va
             best_model_path = save_dir + "best.pt"
             torch.save(leaf_model.state_dict(), best_model_path)

        # if (epoch + 1) % 10 == 0:
        #     torch.save(leaf_model.state_dict(), save_dir + str(epoch) + ".pt")
        #     print(f'{str(epoch+1)}.pt is saved successfully!')

    log_file.close()

    epochs = range(1, len(train_loss) + 1)

    plt.figure(1)
    plt.plot(epochs, train_loss, 'y', label='Training loss')
    plt.plot(epochs, valid_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir + "loss.jpg")

    plt.figure(2)
    plt.plot(epochs, train_acc, 'g', label='Training accuracy')
    plt.plot(epochs, valid_acc, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir + "accuracy.jpg")