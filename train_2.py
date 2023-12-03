import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
import torchvision.transforms as transforms
from loss_function import TripletLoss
from models import SwinTransformer
from dataset import LeafDataset
import matplotlib.pyplot as plt
from models import *
from dataset import TripletLeafDataset
import torch.nn.functional as F

# 命令行传参
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--num_of_epoch', type = int, default=1, help='num of epoch')
parser.add_argument('--batch_size', type = int, default=8, help='batch size')
parser.add_argument('--lr', type = float, default=0.0001, help='learning rate')
parser.add_argument('--train_label_dir', default='/home/hbenke/Project/Lvwc/Project/cv/leaf_disease_classifier/dataset/train.csv', help='train label dir')
parser.add_argument('--train_image_dir', default='/home/hbenke/Project/Lvwc/Project/Data/leaf_disease/train/images/', help='train image dir')
parser.add_argument('--val_label_dir', default='/home/hbenke/Project/Lvwc/Project/cv/leaf_disease_classifier/dataset/val.csv', help='val label dir')
parser.add_argument('--val_image_dir', default='/home/hbenke/Project/Lvwc/Project/Data/leaf_disease/val/images/', help='val image dir')
parser.add_argument('--save_dir', default='/home/hbenke/Project/Lvwc/Project/cv/leaf_disease_classifier/runs/train_result/exp12/', help='save dir')
parser.add_argument('--weights_dir', default='', help='weights path')
args = parser.parse_args()

# 超参数
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

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 获取数据集和dataloader
transform = transforms.Compose([
    transforms.Resize([448, 1120]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = TripletLeafDataset(csv_file=train_label_dir, imgs_path=train_image_dir,
                transform=torchvision.transforms.Compose([
                    # [312, 1000]
                    torchvision.transforms.Resize([224, 224]),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
                    ]))
val_ds = TripletLeafDataset(csv_file=val_label_dir, imgs_path=val_image_dir,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.Resize([224, 224]),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
                    ]))

train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=True, num_workers=4)

TRAIN_SIZE = len(train_ds)
VALID_SIZE = len(val_ds)

# triplet loss
triplet_loss = TripletLoss(margin=1.0)

# 训练函数
def train_fn_triplet(net, loader, optimizer):
    tr_loss = 0
    correct_count = 0
    total_triplets = 0
    items = enumerate(loader)
    total_items = len(loader)

    for _, batch in tqdm(items, total=total_items, desc="train"):
        anchor_images, positive_images, negative_images = batch["anchor"].to(device), batch["positive"].to(device), batch["negative"].to(device)

        optimizer.zero_grad()

        anchor_embeddings = net(anchor_images)
        positive_embeddings = net(positive_images)
        negative_embeddings = net(negative_images)

        loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()

        # 计算三元组排序准确率
        distances_positive = F.pairwise_distance(anchor_embeddings, positive_embeddings)
        distances_negative = F.pairwise_distance(anchor_embeddings, negative_embeddings)
        correct_count += torch.sum(distances_positive < distances_negative).item()
        total_triplets += anchor_embeddings.size(0)

    accuracy = correct_count / total_triplets if total_triplets > 0 else 0
    return accuracy, tr_loss / TRAIN_SIZE

# 验证函数
def valid_fn(net, loader):
    correct_count = 0
    total_triplets = 0
    valid_loss = 0
    items = enumerate(loader)
    total_items = len(loader)
    net.eval()

    with torch.no_grad():
        for _, batch in tqdm(items, total=total_items, desc="val"):
            anchor_images, positive_images, negative_images = batch["anchor"].to(device), batch["positive"].to(device), batch["negative"].to(device)

            anchor_embeddings = net(anchor_images)
            positive_embeddings = net(positive_images)
            negative_embeddings = net(negative_images)

            loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            valid_loss += loss.item()

            # 计算三元组排序准确率
            distances_positive = F.pairwise_distance(anchor_embeddings, positive_embeddings)
            distances_negative = F.pairwise_distance(anchor_embeddings, negative_embeddings)
            correct_count += torch.sum(distances_positive < distances_negative).item()
            total_triplets += anchor_embeddings.size(0)

    accuracy = correct_count / total_triplets if total_triplets > 0 else 0
    return accuracy, valid_loss / VALID_SIZE


# 定义模型
leaf_model = AwesomeModel(num_classes = 6).to(device)

optimizer_triplet = optim.SGD(leaf_model.parameters(), lr=learning_rate, momentum=0.9)

train_loss = []
valid_loss = []
train_acc = []
valid_acc = []


log_file_path = os.path.join(save_dir, "train.log")

if __name__ == "__main__":

    best_val_acc = 0.
    val_predictions = []

    log_file = open(log_file_path, "w")

    # 训练
    for epoch in range(num_of_epoch):

        os.system(f'echo \"Epoch {epoch+1}\"')
    
        leaf_model.train()
        
        ta, tl = train_fn_triplet(leaf_model, loader=train_loader, optimizer=optimizer_triplet)
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

    log_file.close()

    epochs = range(1, len(train_loss) + 1)

    plt.figure(1)
    plt.figure(figsize=(10, 10))
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
    plt.figure(figsize=(10, 10))
    plt.plot(epochs, train_acc, 'g', label='Training accuracy')
    plt.plot(epochs, valid_acc, 'b', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.yticks([i * 0.05 for i in range(int(1 / 0.05) + 1)])
    plt.savefig(save_dir + "accuracy.jpg")