from torch import nn

def weights_init(m):
    """custom weights initialization called on model_generator and model_discriminator"""
    
    classname = m.__class__.__name__
    if classname.find('ConvTranspose2d') != -1 or classname.find('Conv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)