from torch import nn


class ConvTNormRelu(nn.Module):
    """Module which performs a Convolution-transpose, BatchNorm, and ReLU sequentially"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        conv_bias: bool = False,
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
        """
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=conv_bias,
        )
        self.bn = nn.BatchNorm2d(num_features=bias_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass through the module

        Args:
            x: Input data
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvNormLRelu(nn.Module):
    """Module which performs a Convolution, BatchNorm, and LeakyReLU sequentially"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        leaky_slope: float = 0.2,
        conv_bias: bool = False,
    ):
        """Initialize the module layers

        Args:
            in_channels: Number of input channels to the Conv2D
            out_channels: Number of output channels in the feature map after the Conv2D
            bias_channels: Number of channels passed to BatchNorm2D; should be same as Conv2D output
            kernel_size: Size of the Conv2D kernel
            stride: Stide of the ConvTranspose
            padding: Padding of the ConvTranspose
            leaky_slope: Negative slope of the leaky relu; set to 0.2 in the DCGAN paper
            conv_bias: Whether to use a bias in Conv2D; typically this is false if BatchNorm is the following layer
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=conv_bias,
        )
        self.bn = nn.BatchNorm2d(num_features=bias_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope)

    def forward(self, x):
        """Forward pass through the module

        Args:
            x: Input data
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky_relu(x)
        return x


class DCGenerator(nn.Module):
    """Generator for Deep Convolutional GAN (DCGAN)"""

    def __init__(
        self,
        input_vec_ch,
        out_ch_scaler: int = 64,
        out_ch_multiplier: list[int] = [8, 4, 2, 1],
        out_image_ch=3,
    ):
        """Intialize generator layers

        Args:
            input_vec_ch: Number of channels in the input vector; the input vector will be random noise (B, input_vec_ch, 1, 1)
            out_scaler: Scaler to multiply by to get the num of out channels after each convolution
            out_multiplier: Multiplier for each convolutional layer; will be multiplied by out_scaler
            out_image_ch: Number of channels in the output image; should be set to 3 for RGB images
        """
        super().__init__()

        # Multiplier scalar by list to get the number of intermediate output channels
        _out_ch = out_ch_scaler * out_ch_multiplier

        self.tanh = nn.Tanh()

        self.conv_block1 = ConvTNormRelu(
            in_channels=input_vec_ch,
            out_channels=_out_ch[0],
            bias_channels=_out_ch[0],
            kernel_size=4,
            stride=1,
            padding=0,
        )
        self.conv_block2 = ConvTNormRelu(
            in_channels=_out_ch[0],
            out_channels=_out_ch[1],
            bias_channels=_out_ch[1],
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.conv_block3 = ConvTNormRelu(
            in_channels=_out_ch[1],
            out_channels=_out_ch[2],
            bias_channels=_out_ch[2],
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.conv_block4 = ConvTNormRelu(
            in_channels=_out_ch[2],
            out_channels=_out_ch[3],
            bias_channels=_out_ch[3],
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.conv_trans_out = nn.ConvTranspose2d(
            in_channels=_out_ch[3],
            out_channels=out_image_ch,
            kernel_size=4,
            stride=2,
            padding=1,
        )

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
        x = self.tanh(x)

        return x


class DCDiscriminator(nn.Module):
    """Discriminator for Deep Convolutional GAN (DCGAN)

    We do not need an MLP as the last layer because the final feature map will be a 1x1,
    therefore, we can use this scalar value as the prediction. The output shape will be (B, 1, 1, 1)
    so we will need to change the view to match the labels, i.e. output.view(-1)
    """

    def __init__(
        self,
        input_image_ch: int = 3,
        out_ch_scaler: int = 64,
        out_ch_multiplier: list[int] = [1, 2, 4, 8],
    ):
        super().__init__()

        # Multiplier scalar by list to get the number of intermediate output channels
        _out_ch = out_ch_scaler * out_ch_multiplier

        # First layer does not have BatchNorm after according to the paper
        self.conv1 = nn.Conv2d(
            in_channels=input_image_ch,
            out_channels=_out_ch[0],
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.conv_block1 = ConvNormLRelu(
            in_channels=_out_ch[0],
            out_channels=_out_ch[1],
            bias_channels=_out_ch[1],
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.conv_block2 = ConvNormLRelu(
            in_channels=_out_ch[1],
            out_channels=_out_ch[2],
            bias_channels=_out_ch[2],
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.conv_block3 = ConvNormLRelu(
            in_channels=_out_ch[2],
            out_channels=_out_ch[3],
            bias_channels=_out_ch[3],
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=_out_ch[3],
            out_channels=1,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass through the Discriminator.

        Args:
            x: Generated image from the generator; (B, C, H, W)

        Returns:
            Feature map of size (B, 1, 1, 1)
        """
        x = self.conv1(x)
        x = self.leaky_relu(x)

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = self.conv2(x)
        x = self.sigmoid(x)
        return x 


class DCGAN:
    # TODO

    def __init__(self, generator):
        self.generator = generator
