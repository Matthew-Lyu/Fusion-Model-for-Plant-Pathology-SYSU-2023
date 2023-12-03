import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from typing import Any, List, Tuple
import numpy as np

class _DenseLayer(nn.Module):
    def __init__(
        self,
        block_idx:int,layer_idx:int,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
    ) -> None:
        super(_DenseLayer, self).__init__()
        
        self.block_idx = block_idx
        self.layer_idx = layer_idx
        
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False))
        
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False))
    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1) 
        # 1x1卷积，图像的维度变成 128,得到 torch.size（batch,128,height,width)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
            if isinstance(input, Tensor):
                prev_features = [input]
            else:
                prev_features = input
            # 之前的特征图经过 BN1->Relu1->Conv1
            bottleneck_output = self.bn_function(prev_features)
            new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
            return new_features
		
class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        block_idx:int,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
    ) -> None:
        super(_DenseBlock, self).__init__()
        self.block_idx = block_idx
        # 在DenseLayer中输出是相同的，但是输入的维度有来自前面的特征，所以每次输出的维度都是增长的，且增长的速率和输出的维度有关，称为 growth_rate
        for i in range(num_layers):
            layer = _DenseLayer(block_idx=self.block_idx,layer_idx=i,
                num_input_features=num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size
            )
            # 在初始化的时候会形成很多子模型
            self.add_module('denselayer%d' % (i + 1), layer)
            
    def forward(self, init_features: Tensor) -> Tensor:
        # 初始的特征 转换成列表的形式，比如第一个是 torch.size = (batch,64,56,56) - > features=[(batch,64,56,56)]
        features = [init_features]
        # 遍历所有的Layer
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)
	
class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        num_classes: int = 1000,
    ) -> None:

        super(DenseNet, self).__init__()

        # First convolution 输入(3,224,224) -> (64,56,56)
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock 第一次进入num_feature = 64
        num_features = num_init_features

        # 总共创建4个DenseBlock,第1个DenseBlock有6个DenseLayer,第2个DenseBlock有12个DenseLayer,第3个DenseBlock有24个DenseLayer,第4个DenseBlock有16个DenseLayer
        # 每个DenseLayer 有两次卷积
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(block_idx=i,
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
            )
            # 添加到模型当中 
            self.features.add_module('denseblock%d' % (i + 1), block)

            num_features = num_features + num_layers * growth_rate

            # 判断是否执行完的DenseBlock
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)

                # 添加到 features的子模型中
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        # 最终得到 
        # Final batch norm，最后的BN层
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer 
        self.classifier = nn.Linear(num_features, num_classes)


    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
