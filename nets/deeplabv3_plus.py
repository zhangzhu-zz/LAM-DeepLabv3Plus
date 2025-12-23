import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2

#### 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, dim_out, gamma=2, b=1):
        super(ChannelAttention, self).__init__()
        # 自适应核大小计算，基于通道数 C，使用简单的对数函数
        kernel_size = int(abs((math.log2(dim_out) + b) / gamma))
        kernel_size = max(kernel_size, 3)  # 确保核大小至少为3
        if kernel_size % 2 == 0:  # 确保核大小为奇数
            kernel_size += 1
        self.kernel_size = kernel_size  # 保存到 self.kernel_size
        # 池化层
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 动态调整输入以适应池化核大小
        avg_out = self.gap(x)
        max_out = self.gmp(x)

        # 调整维度以适应 nn.Conv1d
        avg_out = avg_out.squeeze(-1).squeeze(-1).unsqueeze(1)  # [batch_size, channels, 1, 1] -> [batch_size, channels] -> [batch_size, 1, channels]
        max_out = max_out.squeeze(-1).squeeze(-1).unsqueeze(1)  # [batch_size, channels, 1, 1] -> [batch_size, channels] -> [batch_size, 1, channels]
        
        # 1D 卷积
        avg_out = self.conv(avg_out)  # [batch_size, 1, channels]
        max_out = self.conv(max_out)  # [batch_size, 1, channels]

        # 相加并通过 Sigmoid
        out = avg_out + max_out
        out = self.sigmoid(out)  # [batch_size, 1, channels]

        # 调整回 [batch_size, channels, 1, 1]
        out = out.squeeze(1).unsqueeze(-1).unsqueeze(-1)
        return out

# 改进自适应SAM模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()

        # 卷积层用于从连接的平均池化和最大池化特征图中学习空间注意力权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)  # 3x3 卷积
        self.conv3 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)  # 7x7 卷积
        self.weight1 = nn.Parameter(torch.ones(1))  # 可学习权重
        self.weight3 = nn.Parameter(torch.ones(1))
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)  # [batch_size, 1, H, W]
        max_out = x.max(dim=1, keepdim=True)[0]  # [batch_size, 1, H, W]
        x = torch.cat([avg_out, max_out], dim=1)  # [batch_size, 2, H, W]

        # 多尺度卷积
        out1 = self.conv1(x)  # 3x3
        out3 = self.conv3(x)  # 7x7

        # 动态加权融合
        Ms = self.weight1 * out1 + self.weight3 * out3
        Ms = self.sigmoid(Ms) # 使用sigmoid激活函数计算注意力权重
        return Ms  

#### 定义CBAM模块的类
class cbam_block(nn.Module):
    def __init__(self, dim_out, kernel_size=7, gamma=2, b=1):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(dim_out, gamma, b)
        self.spatialattention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 通道注意力
        Mc = self.channelattention(x)  # [batch_size, in_channels, 1, 1]
        x = x * Mc  # 通道加权 [batch_size, in_channels, H, W]

        # 空间注意力
        Ms = self.spatialattention(x)  # [batch_size, 1, H, W]
        x = x * Ms  # 空间加权 [batch_size, in_channels, H, W]
        return x
    
### 定义改进的ECA注意力模块的类
class eca_block(nn.Module):
    def __init__(self, channel,r=16, b=1, gamma=2):
        super(eca_block, self).__init__()
        # 计算1D卷积的核大小
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = max(kernel_size, 3)
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False)
        )

        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  
        # 全局平均池化
        avg_out = self.gap(x)  # 形状 (b, c)
        
        # SE 模块：降维后升维
        y = avg_out.squeeze(-1).permute(0, 2, 1)
        y = self.fc(y)  # 形状 (b, c)
        y = self.sigmoid(y)
        # ECA 模块：调整形状为 (b, 1, c)，应用1D卷积
        y = self.conv(y)  # 形状 (b, 1, c)
        # Sigmoid 激活，生成注意力权重
        y = self.sigmoid(y)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        
        # 将权重传入到输入特征图
        return x * y.expand_as(x)

    
# Strip Pooling
class StripPooling(nn.Module):
    def __init__(self, in_channels, out_channels, bn_mom=0.1):
        super(StripPooling, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_v = nn.AdaptiveAvgPool2d((1, None))
        self.conv_h = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv_v = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.size()
        h_pool = self.pool_h(x)
        h_pool = self.conv_h(h_pool)
        h_pool = F.interpolate(h_pool, (h, w), mode='bilinear', align_corners=True)
        v_pool = self.pool_v(x)
        v_pool = self.conv_v(v_pool)
        v_pool = F.interpolate(v_pool, (h, w), mode='bilinear', align_corners=True)
        out = h_pool + v_pool
        out = self.bn(out)
        out = self.relu(out)
        return out

class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model           = mobilenetv2(pretrained)
        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x 

#-----------------------------------------#
#   ASPP特征提取模块
#   利用不同膨胀率的膨胀卷积进行特征提取
#-----------------------------------------#
class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch5 = StripPooling(dim_in, dim_out, bn_mom=bn_mom)

		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
                cbam_block(dim_out=dim_out, kernel_size=7, gamma=2, b=1),  # 在此添加CBAM
		)

	def forward(self, x):
		[b, c, row, col] = x.size()
        #-----------------------------------------#
        #   一共五个分支
        #-----------------------------------------#
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
        #-----------------------------------------#
        #   第五个分支，条形池化+卷积
        #-----------------------------------------#
		global_feature = self.branch5(x)
        #-----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result

class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, downsample_factor=16):
        super(DeepLab, self).__init__()
        if backbone=="xception":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,256]
            #   主干部分    [30,30,2048]
            #----------------------------------#
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone=="mobilenet":
            #----------------------------------#
            #   获得两个特征层
            #   浅层特征    [128,128,24]
            #   主干部分    [30,30,320]
            #----------------------------------#
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception.'.format(backbone))

        #-----------------------------------------#
        #   ASPP特征提取模块
        #   利用不同膨胀率的膨胀卷积进行特征提取
        #-----------------------------------------#
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16//downsample_factor)
        
        #----------------------------------#
        #   浅层特征边
        #----------------------------------#
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            eca_block(channel=48),  # ECANet 在 1x1 卷积后
        )		

        self.cat_conv = nn.Sequential(
            nn.Conv2d(48+256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        #-----------------------------------------#
        #   获得两个特征层
        #   low_level_features: 浅层特征-进行卷积处理
        #   x : 主干部分-利用ASPP结构进行加强特征提取
        #-----------------------------------------#
        low_level_features, x = self.backbone(x)
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)
        
        #-----------------------------------------#
        #   将加强特征边上采样
        #   与浅层特征堆叠后利用卷积进行特征提取
        #-----------------------------------------#
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

