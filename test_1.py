import argparse
import math
import os
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from loss_function.ArcFaceLoss import ArcFaceLoss
from loss_function.FocalLoss import FocalLoss
import numpy as np
import torch.nn.functional as F


from models import *
from dataset import LeafDataset
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

exp_number = 'exp1'

model_save_root_path = f'/home/hbenke/Project/Lvwc/Project/cv/leaf_disease_classifier/runs/train_result/{exp_number}/best.pt'

save_dir = save_dir+f'{exp_number}/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 获取数据集和dataloader
test_ds = LeafDataset(csv_file=test_label_dir, imgs_path=test_image_dir,
                transform=torchvision.transforms.Compose([
                    # [312, 1000]
                    torchvision.transforms.Resize([224, 224]),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],),
                    ]))

test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True, num_workers=4) # num_workers=4 表示用四个子进程加载数据

TEST_SIZE = len(test_ds)

# 二分类交叉熵损失
loss_fn = torch.nn.BCEWithLogitsLoss()

model_name = 'AwesomeModelwithTL'

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

leaf_model.load_state_dict(torch.load(model_save_root_path))
leaf_model.eval()

csv_path_txt = f'/home/hbenke/Project/Lvwc/Project/cv/leaf_disease_classifier/runs/test_result/{exp_number}/{exp_number}.txt'
f = open(csv_path_txt, 'w')
f.write('healthy,scab,frog_eye_leaf_spot,rust,complex,powdery_mildew,healthy,scab,frog_eye_leaf_spot,rust,complex,powdery_mildew\n')

def print_log(pred, label):
    for i in range(0, len(pred)):
        for j in range(0, len(pred[0])):
            if pred[i][j] < 0.5:
                pred[i][j] = 0
            else:
                pred[i][j] = 1
    for i in range(0, len(pred)):
        msg = f'{pred[i][0]},{pred[i][1]},{pred[i][2]},{pred[i][3]},{pred[i][4]},{pred[i][5]}, {label[i][0]},{label[i][1]},{label[i][2]},{label[i][3]},{label[i][4]},{label[i][5]}\n'
        f.write(msg)

# 打开存储的Log文件
path = save_dir + 'test.log'
fa = open(path, 'w')

def Eval(net, loader):
    valid_loss = 0
    valid_accuracy = 0 
    with torch.no_grad():          
        # 创建一个迭代对象
        items = enumerate(loader)
        total_items = len(loader)  # 获取迭代对象的总长度    
        for _, (images, labels) in tqdm(items, total=total_items, desc="val"):       
            images, labels = images.to(device), labels.to(device)
            net.eval()
            predictions = net(images)
            loss = loss_fn(predictions, labels.squeeze(-1))       
            valid_loss += loss.item()
            batch_shape = list(predictions.size())
            print_log(predictions.detach().cpu().numpy().tolist(), labels.detach().cpu().numpy().tolist())
            for i in range(batch_shape[0]):
                for j in range(batch_shape[1]):
                    prediction = 1 if predictions.detach().cpu().numpy()[i][j] >= 0.5 else 0
                    if prediction == labels.detach().cpu().numpy()[i][j]:
                        valid_accuracy += 1.0 / batch_shape[1]
    return valid_accuracy/TEST_SIZE, valid_loss/TEST_SIZE


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

        for _, (images, labels) in tqdm(items, total=total_items, desc="Calculate f1 curve"):
            images, labels = images.to(device), labels.to(device)
            predictions = torch.sigmoid(net(images))  # 应用sigmoid确保预测在[0, 1]范围内

            for class_idx in range(num_classes):
                all_predictions[class_idx].extend(predictions[:, class_idx].detach().cpu().numpy().flatten())
                all_labels[class_idx].extend(labels[:, class_idx].detach().cpu().numpy().flatten())

    plt.figure(figsize=(10, 10))

    for class_idx in range(num_classes):
        precision, recall, thresholds = precision_recall_curve(all_labels[class_idx], all_predictions[class_idx])

        f1_scores = [f1_score(all_labels[class_idx], [1 if p >= threshold else 0 for p in all_predictions[class_idx]]) for threshold in thresholds]
        auc_value = auc(recall, precision)

        class_str = str(class_idx)

        plt.plot(thresholds, f1_scores, label=f'{class_indices[class_str]} (AUC={auc_value:.2f})')

    plt.plot(thresholds, f1_scores, label=f'F1 Curve (AUC={auc_value:.2f})', linewidth = 2, color = 'r')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Curve')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, 'precision_recall_curve.png')
    plt.savefig(save_path)
    print(f'F1 curve saved at {save_dir}')

if __name__ == "__main__":
    leaf_model.eval()
    test_acc, test_loss = Eval(leaf_model, loader=test_loader)
    print(f'test_acc: {test_acc}, test_loss: {test_loss}')
    fa.write(f'{exp_number},{model_name},{test_acc},{test_loss}\n')
    calculate_f1_curve_and_save_per_class(leaf_model, loader=test_loader, save_dir=save_dir, num_classes=6)