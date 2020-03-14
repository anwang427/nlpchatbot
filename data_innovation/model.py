from torch.distributions import Categorical
import torch.nn as nn
import numpy as np
import math

import model_helpers

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
    elif isinstance(m, nn.Conv2d):
        n_weights = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        nn.init.normal_(m.weight, mean=0., std=math.sqrt(2. / n_weights))

class ConvUnit(nn.Module):
    def __init__(self, img_shape, out_channels, conv_kernel_size, conv_stride, conv_padding, maxpool_kernel_size, maxpool_stride):
        super(ConvUnit, self).__init__()

        height, width, num_channels = img_shape

        self.unit = nn.Sequential(
            nn.Conv2d(num_channels, out_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=conv_padding),
            nn.MaxPool2d(maxpool_kernel_size, stride=maxpool_stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        return self.unit(input)


class SeedlingVision(nn.Module):
    """
    Input to Conv2d is of shape (Batch Size, Number Channels, Height, Width)
    Even grayscale requires 4 dimensions.
    """
    def __init__(self, input_shape):
        assert len(img_shape) == 3
        super(ActorCritic, self).__init__()

        self.img_shape = img_shape
        self.input_shape = None
        self.output_shape = None
        height, width, num_channels = input_shape

        # Actor Neural Networks ------------------------
        # conv_unit_pars: out_channels, conv_kernel_size, conv_stride, conv_padding, maxpool_kernel_size, maxpool_stride
        img_pars = [16, 5, 1, 2, 4, 4]
        img_output_shape = get_cu_output_shape(img_shape, img_pars)
        comb_pars1 = [8, 5, 1, 0, 2, 1]
        comb_output_shape1 = get_cu_output_shape(img_output_shape, comb_pars1)
        # comb_pars2 = [8, 4, 1, 0, 2, 2]
        # comb_output_shape2 = get_cu_output_shape(comb_output_shape1, comb_pars2)
        self.nn = nn.Sequential(
            ConvUnit(
                img_shape, 
                out_channels=img_pars[0], 
                conv_kernel_size=img_pars[1], 
                conv_stride=img_pars[2], 
                conv_padding=img_pars[3], 
                maxpool_kernel_size=img_pars[4],
                maxpool_stride=img_pars[5])      
            nn.Flatten(),
            nn.Linear(np.prod(comb_output_shape1), 64),
            nn.Linear(64, 32),
            nn.Linear(32, num_outputs),
            nn.LogSoftmax()
        )
        

        self.apply(init_weights)
        
    def forward(self, img):
        """Performs forward pass for the NN. Importantly, the input reshaping is done here
        
        Args:
            img (tensor): Batched 3D image. 
                4D tensor (Batch Size, , Height, Width, Number Channels)
                Height, width, and number of channels much match the input shape given in the initialization of the neural net
        
        Returns:
            tensor: Action distribution and the value of the state
        """

        return self.forward(img)

