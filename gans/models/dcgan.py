from torch import nn
from torch.nn import Functional as F


################## START HERE, GO THROUGH GENERATOR AND ALSO NEED TO TEST DATASET #####################
class ConvTNormRelu(nn.Module):
    """Module which performs a Convolution-transpose, BatchNorm, and ReLU sequentially"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding:int = 0,
        conv_bias: bool = False,
        norm_bias: bool = True,
    ):
        """Initialize the module layers

        Args:
            in_channels: Number of input channels to the Conv2D
            out_channels: Number of output channels in the feature map after the Conv2D
            bias_channels: Number of channels passed to BatchNorm2D; should be same as Conv2D output
            kernel_size: Size of the Conv2D kernel
            stride: Stide of the ConvTranspose
            padding: Padding of the ConvTranspose
            conv_bias: Whether to use a bias in Conv2D; typically this is false if BatchNorm is the following layer
            bias_norm: Whether to use a bias in BatchNorm2D; typically this is true with a preceeding Conv layer
        """
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels=out_channels, kernel_Size=kernel_size, stride=stride, padding=padding, bias=conv_bias
        )
        self.bn = nn.BatchNorm2d(num_features=bias_channels, bias=norm_bias)

    def forward(self, x):
        """Forward pass through the module
        
        Args:
            x: Input data
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.ReLU(x)
        return x


class DCGenerator(nn.Module):
    """Generator for DCGAN"""

    def __init__(self, input_vec_ch, out_scaler: int=64, out_multiplier: list[int] = [8, 4, 2, 1], out_image_ch=3):
        """Intialize generator layers
        
        Args:
            input_vec_ch: Number of channels in the input vector
            out_scaler: Scaler to multiply by to get the num of out channels after each convolution
            out_multiplier: Multiplier for each convolutional layer; will be multiplied by out_scaler
            out_image_ch: Number of channels in the output image; should be set to 3 for RGB images 
        """
        # Multiplier scalar by list to get the number of intermediate output channels
        _out_ch = out_scaler*out_multiplier

        self.conv_block1 = ConvTNormRelu(in_channels=input_vec_ch, out_channels=_out_ch[0], bias_channels=_out_ch[0], kernel_size=4, stride=1, padding=0)
        self.conv_block2 = ConvTNormRelu(in_channels=_out_ch[0], out_channels=_out_ch[1], bias_channels=_out_ch[1], kernel_size=4, stride=2, padding=1)
        self.conv_block3 = ConvTNormRelu(in_channels=_out_ch[1], out_channels=_out_ch[2], bias_channels=_out_ch[2], kernel_size=4, stride=2, padding=1)
        self.conv_block4 = ConvTNormRelu(in_channels=_out_ch[2], out_channels=_out_ch[3], bias_channels=_out_ch[3], kernel_size=4, stride=2, padding=1)
        self.conv_trans_out = nn.ConvTranspose2d(in_channels=_out_ch[3], out_channels=out_image_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        """Forward pass through the Generator.
        
        Args:
            x: input vector

        Returns:
            Generated image
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = self.conv_trans_out(x)

        # Return the data into the range [-1, 1]
        x = F.Tanh(x)
        
        return x

