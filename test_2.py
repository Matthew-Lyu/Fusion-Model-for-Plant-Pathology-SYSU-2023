import argparse
import math
import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from loss_function import TripletLoss
import numpy as np
import torch.nn.functional as F

from models import *
from dataset import LeafDataset, TripletLeafDataset
from sklearn.metrics import precision_recall_curve, f1_score, auc
from scipy.interpolate import interp1d
import json

# 命令行传参
parser = argparse.ArgumentParser()
parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--batch_size', type = int, default=8, help='batch size')
parser.add_argument('--lr', type = float, default=0.0001, help='learning rate')
parser.add_argument('--test_label_dir', default='/home/hbenke/Project/Lvwc/Project/cv/leaf_disease_classifier/dataset/test.csv', help='test label dir')
parser.add_argument('--test_image_dir', default='/home/hbenke/Project/Lvwc/Project/Data/leaf_disease/test/images/', help='test image dir')
parser.add_argument('--weights_dir', default='', help='weights path')
parser.add_argument('--save_dir', default='/home/hbenke/Project/Lvwc/Project/cv/leaf_disease_classifier/runs/test_result/', help='save path')
args = parser.parse_args()

# 超参数
# 设置为gpu训练
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

learning_rate = args.lr
batch_size = args.batch_size

# 数据路径
test_label_dir = args.test_label_dir
test_image_dir = args.test_image_dir
weights_dir = args.weights_dir
save_dir = args.save_dir

exp_number = 'exp11'

model_save_root_path = f'/home/hbenke/Project/Lvwc/Project/cv/leaf_disease_classifier/runs/train_result/{exp_number}/best.pt'

save_dir = save_dir+f'{exp_number}/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 获取数据集和dataloader
test_ds = TripletLeafDataset(csv_file=test_label_dir, imgs_path=test_image_dir,
                transform=torchvision.transforms.Compose([
                    # [312, 1000]
                    torchvision.transforms.Resize([224, 224]),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
                    ]))

test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True, num_workers=4) # num_workers=4 表示用四个子进程加载数据

TEST_SIZE = len(test_ds)

triplet_loss = TripletLoss(margin=1.0)

model_name = 'AwesomeModelwithTL'

# leaf_model = ResNet50(num_classes=6, pretrained=True).to(device)

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

leaf_model = AwesomeModel4(num_classes=6).to(device)

# leaf_model = AwesomeModel2(num_classes=6).to(device)

# leaf_model = AwesomeModel3(num_classes=6).to(device)

# leaf_model = AwesomeModel4(num_classes=6).to(device)

leaf_model.load_state_dict(torch.load(model_save_root_path))
leaf_model.eval()

# 打开存储的Log文件
path = save_dir + 'test.log'
f = open(path, 'w')


# 验证函数
def Eval(net, loader):
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
    return accuracy, valid_loss / TEST_SIZE


# 读取 class_indices.json
json_file_path = '/home/hbenke/Project/Lvwc/Project/cv/leaf_disease_classifier/class_indices.json'
with open(json_file_path, 'r') as file:
    class_indices_str = file.read()

class_indices = json.loads(class_indices_str)

def calculate_f1_curve_and_save_per_class(net, loader, save_dir, num_classes):
    net.eval()
    all_predictions = [[] for _ in range(num_classes)]
    all_labels = [[] for _ in range(num_classes)]

    with torch.no_grad():
        items = enumerate(loader)
        total_items = len(loader)

        for _, batch in tqdm(items, total=total_items, desc="Calculate f1 curve"):
            anchor_images, positive_images, negative_images = batch["anchor"].to(device), batch["positive"].to(device), batch["negative"].to(device)
            anchor_embeddings = net(anchor_images)

            predictions = torch.sigmoid(anchor_embeddings)  # Apply sigmoid to ensure predictions are in [0, 1] range

            # Since labels are not directly provided in the batch, use a fixed threshold for F1 calculation
            threshold = 0.5  # You can adjust this threshold based on your needs

            for class_idx in range(num_classes):
                all_predictions[class_idx].extend(predictions[:, class_idx].detach().cpu().numpy().flatten())
                all_labels[class_idx].extend((predictions[:, class_idx] > threshold).int().cpu().numpy().flatten())

    plt.figure(figsize=(10, 10))

    for class_idx in range(num_classes):
        precision, recall, _ = precision_recall_curve(all_labels[class_idx], all_predictions[class_idx])

        f1_value = f1_score(all_labels[class_idx], [1 if p >= threshold else 0 for p in all_predictions[class_idx]])
        auc_value = auc(recall, precision)

        class_str = str(class_idx)

        plt.plot(recall, precision, label=f'{class_str} (F1={f1_value:.2f}, AUC={auc_value:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, 'precision_recall_curve.png')
    plt.savefig(save_path)
    print(f'Precision-Recall curve saved at {save_path}')


if __name__ == "__main__":
    leaf_model.eval()
    test_acc, test_loss = Eval(leaf_model, loader=test_loader)
    print(f'test_acc: {test_acc}, test_loss: {test_loss}')
    f.write(f'{exp_number},{model_name},{test_acc},{test_loss}\n')
    #calculate_f1_curve_and_save_per_class(leaf_model, loader=test_loader, save_dir=save_dir, num_classes=6)