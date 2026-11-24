import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = torch.nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class Dense_Block(nn.Module):
    def __init__(self, in_features, out_features):
        super(Dense_Block, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = torch.nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(out_features)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * -ctx.lambd), None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

class HyperspectralGenerator(nn.Module):
    """高光谱数据生成器，基于PWDA的ResBaseNonePool架构"""
    def __init__(self, in_channels, patch_size=15):
        super(HyperspectralGenerator, self).__init__()
        self.patch_size = patch_size
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear1 = nn.Linear(1536, 1024)
        # 输入层：将高光谱数据映射到3通道
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, 3, (3, 3), 1, 1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        
        # 3D卷积分支（处理高光谱维度）
        self.convf1 = nn.Conv3d(1, 64, kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=True)
        self.bnf1 = nn.BatchNorm3d(64)
        self.activation1 = nn.ReLU()
        
        # Residual block 1
        self.convf2 = nn.Conv3d(64, 64, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), bias=True)
        self.bnf2 = nn.BatchNorm3d(64)
        self.activation2 = nn.ReLU()
        self.convf3 = nn.Conv3d(64, 64, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), bias=True)
        self.bnf3 = nn.BatchNorm3d(64)
        self.activation3 = nn.ReLU()
        
        # 完成层
        self.convf4 = nn.Conv3d(64, 512, kernel_size=(7, 1, 1), stride=1, padding=(3, 0, 0), bias=True)
        self.bnf4 = nn.BatchNorm3d(512)
        self.activation4 = nn.ReLU()
        
        # 最终卷积层
        self.convf5 = nn.Conv3d(512, 1024, kernel_size=(((in_channels - 7) // 2 + 1), 1, 1), bias=True)
        self.bnf5 = nn.BatchNorm3d(1024)
        self.activation5 = nn.ReLU()
        
        # 2D ResNet分支（简化版）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 简化的ResNet层
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=1)
        self.layer3 = self._make_layer(128, 256, 2, stride=1)
        self.layer4 = self._make_layer(256, 512, 2, stride=1)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(Conv_Block(in_channels, out_channels, 3, stride, 1))
        for _ in range(1, blocks):
            layers.append(Conv_Block(out_channels, out_channels, 3, 1, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
       # [32,105,15,15]
        x1 = x.unsqueeze(1)  # (B, 1, C, H, W)

        x1 = self.convf1(x1) # [32,64,48,15,15]
        x1 = self.activation1(self.bnf1(x1))
        
        # 残差层
        residual = x1
        x1 = self.convf2(x1)
        x1 = self.activation2(self.bnf2(x1))
        x1 = self.convf3(x1)
        x1 = residual + x1
        x1 = self.activation3(self.bnf3(x1))
        
        # 完成层
        x1 = self.convf4(x1) #[32,512,48,15,15]
        x1 = self.activation4(self.bnf4(x1))
        
        x1 = self.convf5(x1)#[32,1024,48,15,15]
        x1 = self.activation5(self.bnf5(x1))
        
        # 重塑为2D
        x1 = x1.reshape(x1.size(0), x1.size(1), x1.size(3), x1.size(4))
        
        # 2D分支处理
        x2 = self.conv_in(x)
        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)
        
        # 拼接两个分支
        x = torch.cat((x1, x2), 1)  # 1024 + 512 = 1536

        x  = self.average_pooling(x ).view(x .size(0), -1)
        x  = self.linear1(x)
        return x

class HyperspectralClassifier(nn.Module):
    """高光谱分类器"""
    def __init__(self, num_classes, input_dim=1024, unit_size=1024):
        super(HyperspectralClassifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, unit_size)
        self.bn1 = nn.BatchNorm1d(unit_size, affine=True, track_running_stats=True)
        self.linear2 = nn.Linear(unit_size, unit_size)
        self.bn2 = nn.BatchNorm1d(unit_size, affine=True, track_running_stats=True)
        self.classifier = nn.Linear(unit_size, num_classes)
        self.drop = nn.Dropout(p=0.3)
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, rois, constant=1, adaption=False, pooling=False, return_feat=False):
        #if pooling:
        #    rois = self.average_pooling(rois).view(rois.size(0), -1)

        #x = self.drop(F.relu(self.bn1(self.linear1(rois))))
        x = self.drop(F.relu(self.bn1(rois)))
        x = self.drop(F.relu(self.bn2(self.linear2(x))))

        # GRL或梯度缩放
        x_rev = grad_reverse(x, constant) if adaption else x
        logits = self.classifier(x_rev)

        if return_feat:
            return logits, x
        else:
            return logits

class Net(nn.Module):
    """OSBP高光谱网络"""
    def __init__(self, in_channels, num_classes, patch_size=15):
        super(Net, self).__init__()
        self.generator = HyperspectralGenerator(in_channels, patch_size)
        self.classifier = HyperspectralClassifier(num_classes, input_dim=1536)


    def forward(self, x, constant=1, adaption=False):
        rois = self.generator(x)

        x = self.classifier(rois, constant, adaption)
        return x
