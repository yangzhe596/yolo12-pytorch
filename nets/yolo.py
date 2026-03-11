#-------------------------------------#
#       YOLOv12 Model
#-------------------------------------#
import os

import numpy as np
import torch
import torch.nn as nn

from .backbone import Conv, C3k2, SPPF, DWConv
from .a2c2f import A2C2f
from .cbam import CBAM
from .yolo_training import weights_init
from utils.utils_bbox import make_anchors


def fuse_conv_and_bn(conv, bn):
    """混合Conv2d + BatchNorm2d 减少计算量"""
    fusedconv = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True
    ).requires_grad_(False).to(conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


class DFL(nn.Module):
    """Distribution Focal Loss (DFL) 模块"""

    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class YoloBody(nn.Module):
    """YOLOv12 模型主体

    支持 n/s/m/l/x 五个版本，使用 Area-Attention 和 C3k2 模块。
    """

    # 模型缩放参数
    depth_dict = {'n': 0.50, 's': 0.50, 'm': 0.50, 'l': 1.00, 'x': 1.00}
    width_dict = {'n': 0.25, 's': 0.50, 'm': 1.00, 'l': 1.00, 'x': 1.50}
    max_channels_dict = {'n': 1024, 's': 1024, 'm': 512, 'l': 512, 'x': 512}

    # 预训练权重下载链接
    PRETRAINED_URLS = {
        'n': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt',
        's': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12s.pt',
        'm': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12m.pt',
        'l': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12l.pt',
        'x': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12x.pt',
    }

    def __init__(self, input_shape, num_classes, phi, pretrained=False, pretrained_path=None):
        """Initialize YOLOv12 model.

        Args:
            input_shape (list): 输入尺寸 [H, W]
            num_classes (int): 类别数量
            phi (str): 模型版本 ('n', 's', 'm', 'l', 'x')
            pretrained (bool): 是否加载预训练权重
            pretrained_path (str): 自定义预训练权重路径
        """
        super(YoloBody, self).__init__()

        self.phi = phi
        self.num_classes = num_classes

        # 获取缩放参数
        dep_mul = self.depth_dict[phi]
        wid_mul = self.width_dict[phi]
        max_channels = self.max_channels_dict[phi]

        # 计算基础通道和深度
        def make_round(x):
            return max(round(x * dep_mul), 1)

        def make_divisible(x, divisor=8):
            return max(int(x // divisor * divisor), divisor)

        base_channels = min(make_divisible(wid_mul * 64), max_channels)
        base_depth = make_round(2)  # 基础深度

        #----------------------- Backbone -----------------------#
        # P1/2: 3, 640, 640 => 64, 320, 320
        self.stem = Conv(3, base_channels, 3, 2)

        # P2/4: 64, 320, 320 => 128, 160, 160
        ch1 = min(make_divisible(wid_mul * 128), max_channels)
        self.dark2 = nn.Sequential(
            Conv(base_channels, ch1, 3, 2),
            C3k2(ch1, ch1, n=make_round(2), c3k=False, e=0.25),
        )

        # P3/8: 128, 160, 160 => 256, 80, 80
        ch2 = min(make_divisible(wid_mul * 256), max_channels)
        self.dark3 = nn.Sequential(
            Conv(ch1, ch2, 3, 2),
            C3k2(ch2, ch2, n=make_round(2), c3k=False, e=0.25),
        )

        # P4/16: 256, 80, 80 => 512, 40, 40
        ch3 = min(make_divisible(wid_mul * 512), max_channels)
        self.dark4 = nn.Sequential(
            Conv(ch2, ch3, 3, 2),
            A2C2f(ch3, ch3, n=make_round(4), a2=True, area=4, residual=False),
        )

        # P5/32: 512, 40, 40 => 1024, 20, 20
        ch4 = min(make_divisible(wid_mul * 1024), max_channels)
        self.dark5 = nn.Sequential(
            Conv(ch3, ch4, 3, 2),
            A2C2f(ch4, ch4, n=make_round(4), a2=True, area=1, residual=False),
        )

        #----------------------- FPN + PAN -----------------------#
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # P4: 1024 + 512 => 512
        self.conv3_for_upsample1 = A2C2f(ch4 + ch3, ch3, n=make_round(2), a2=False)

        # P3: 512 + 256 => 256
        self.conv3_for_upsample2 = A2C2f(ch3 + ch2, ch2, n=make_round(2), a2=False)

        # P4: 256 => 512
        self.down_sample1 = Conv(ch2, ch2, 3, 2)
        self.conv3_for_downsample1 = A2C2f(ch3 + ch2, ch3, n=make_round(2), a2=False)

        # P5: 512 => 1024
        self.down_sample2 = Conv(ch3, ch3, 3, 2)
        self.conv3_for_downsample2 = C3k2(ch4 + ch3, ch4, n=make_round(2), c3k=True)

        #----------------------- CBAM -----------------------#
        self.cbam1 = CBAM(ch2)
        self.cbam2 = CBAM(ch3)
        self.cbam3 = CBAM(ch4)

        #----------------------- Detect Head -----------------------#
        ch = [ch2, ch3, ch4]
        self.nl = len(ch)
        self.reg_max = 16
        self.no = num_classes + self.reg_max * 4

        # 计算stride
        self.stride = torch.tensor([input_shape[0] / x for x in [80, 40, 20]])

        # 检测头
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(num_classes, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1))
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, num_classes, 1),
            )
            for x in ch
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        # 初始化权重
        if not pretrained:
            weights_init(self)

        # 初始化shape用于缓存anchors
        self.shape = None
        self.anchors = None
        self.strides = None

        # 加载预训练权重
        if pretrained:
            self._load_pretrained_weights(pretrained_path)

    def _load_pretrained_weights(self, pretrained_path=None):
        """加载预训练权重"""
        import ssl
        import urllib.request

        if pretrained_path is None:
            # 自动下载预训练权重
            url = self.PRETRAINED_URLS.get(self.phi)
            if url is None:
                print(f"No pretrained weights available for yolo12{self.phi}")
                return

            filename = f"yolo12{self.phi}.pt"
            save_dir = "model_data"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)

            if os.path.exists(save_path):
                print(f"Pretrained weights already exists: {save_path}")
                pretrained_path = save_path
            else:
                print(f"Downloading pretrained weights from {url}")
                try:
                    # 创建不验证SSL的上下文
                    ssl_context = ssl.create_default_context()
                    ssl_context.check_hostname = False
                    ssl_context.verify_mode = ssl.CERT_NONE

                    # 使用urllib下载
                    with urllib.request.urlopen(url, context=ssl_context) as response:
                        with open(save_path, 'wb') as f:
                            f.write(response.read())
                    pretrained_path = save_path
                    print(f"Downloaded to {save_path}")
                except Exception as e:
                    print(f"Failed to download pretrained weights: {e}")
                    print(f"Please manually download from: {url}")
                    print(f"And save to: {save_path}")
                    return

        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location="cpu")

            # 处理不同格式的checkpoint
            if isinstance(checkpoint, dict):
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                    if hasattr(state_dict, 'state_dict'):
                        state_dict = state_dict.state_dict()
                    elif not isinstance(state_dict, dict):
                        state_dict = state_dict.float().state_dict()
                elif 'ema' in checkpoint:
                    state_dict = checkpoint['ema']
                    if hasattr(state_dict, 'float'):
                        state_dict = state_dict.float().state_dict()
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint

            # 加载匹配的权重
            model_dict = self.state_dict()
            load_key, no_load_key, temp_dict = [], [], {}

            for k, v in state_dict.items():
                # 处理key名称差异
                new_k = k
                if k.startswith('model.'):
                    new_k = k[6:]

                if new_k in model_dict.keys() and np.shape(model_dict[new_k]) == np.shape(v):
                    temp_dict[new_k] = v
                    load_key.append(new_k)
                else:
                    no_load_key.append(k)

            model_dict.update(temp_dict)
            self.load_state_dict(model_dict)

            print(f"Successfully loaded {len(load_key)} keys")
            if no_load_key:
                print(f"Failed to load {len(no_load_key)} keys (expected for different num_classes)")

    def fuse(self):
        """融合Conv和BN层以加速推理"""
        print('Fusing layers... ')
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
        return self

    @property
    def backbone(self):
        """Return backbone modules for freeze/unfreeze training"""
        return nn.ModuleList([self.stem, self.dark2, self.dark3, self.dark4, self.dark5])

    def forward(self, x):
        # Backbone
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        feat1 = self.cbam1(x)  # P3: 256, 80, 80
        x = self.dark4(x)
        feat2 = self.cbam2(x)  # P4: 512, 40, 40
        x = self.dark5(x)
        feat3 = self.cbam3(x)  # P5: 1024, 20, 20

        # FPN + PAN
        P5_upsample = self.upsample(feat3)
        P4 = torch.cat([P5_upsample, feat2], 1)
        P4 = self.conv3_for_upsample1(P4)

        P4_upsample = self.upsample(P4)
        P3 = torch.cat([P4_upsample, feat1], 1)
        P3 = self.conv3_for_upsample2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, feat3], 1)
        P5 = self.conv3_for_downsample2(P5)

        # Detect Head
        shape = P3.shape
        x = [P3, P4, P5]

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2).split((self.reg_max * 4, self.num_classes), 1)
        dbox = self.dfl(box)

        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device)
