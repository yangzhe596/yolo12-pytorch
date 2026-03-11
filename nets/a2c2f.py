#-------------------------------------#
#       YOLOv12 A2C2f Module
#       Area-Attention C2f
#-------------------------------------#
import torch
import torch.nn as nn

from .backbone import Conv, C3k
from .attention import ABlock


class A2C2f(nn.Module):
    """Area-Attention C2f module for enhanced feature extraction.

    这是YOLOv12的核心模块，扩展C2f架构并融入区域注意力机制。
    支持area-attention和标准卷积两种模式。

    Attributes:
        cv1 (Conv): 初始1x1卷积层
        cv2 (Conv): 最终1x1卷积层
        gamma (nn.Parameter | None): 可学习的残差缩放参数
        m (nn.ModuleList): ABlock或C3k模块列表
    """

    def __init__(
        self,
        c1,
        c2,
        n=1,
        a2=True,
        area=1,
        residual=False,
        mlp_ratio=2.0,
        e=0.5,
        g=1,
        shortcut=True,
    ):
        """Initialize Area-Attention C2f module.

        Args:
            c1 (int): 输入通道数
            c2 (int): 输出通道数
            n (int): ABlock或C3k模块数量
            a2 (bool): 是否使用area attention，False则使用C3k
            area (int): 特征图划分的区域数量
            residual (bool): 是否使用带可学习gamma参数的残差连接
            mlp_ratio (float): MLP隐藏层扩展比例
            e (float): 通道扩展比例
            g (int): 分组卷积的组数
            shortcut (bool): C3k块是否使用shortcut连接
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through A2C2f layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))

        if self.gamma is not None:
            return x + self.gamma.view(-1, self.gamma.shape[0], 1, 1) * y
        return y
