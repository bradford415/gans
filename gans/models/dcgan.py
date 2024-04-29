from torch import nn
from torch.nn import Functional as F


################## START HERE, GO THROUGH GENERATOR AND ALSO NEED TO TEST DATASET #####################
class ConvTNormRelu(nn.Module):
    """Module which performs a Convolution-transpose, BatchNorm, and ReLU sequentially"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, conv_bias:bool= False, norm_bias=True):
        self.conv = nn.Conv2d(in_channels, out_channels=, bias=conv_bias)
        self.bn = nn.BatchNorm2d(bias=norm_bias)

    def forward(x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.ReLU(x)


class Generator(nn.Module):

    def __init__():