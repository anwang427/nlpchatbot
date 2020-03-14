import numpy as np
import cv2

def get_output_size_1D(input_size, kernel_size, padding, stride):
    return int(np.floor((input_size - kernel_size + 2 * padding) / stride + 1))

def get_cu_output_shape(img_shape, conv_unit_pars):

    input_length, input_width, _ = img_shape
    out_channels, conv_kernel_size, conv_stride, conv_padding, maxpool_kernel_size, maxpool_stride = conv_unit_pars

    # Through the conv layer
    output_length = get_output_size_1D(input_length, conv_kernel_size, conv_padding, conv_stride)
    output_width = get_output_size_1D(input_width, conv_kernel_size, conv_padding, conv_stride)

    # Through the maxpool layer
    output_length = get_output_size_1D(output_length, maxpool_kernel_size, 0, maxpool_stride)
    output_width = get_output_size_1D(output_width, maxpool_kernel_size, 0, maxpool_stride)

    return [output_length, output_width, out_channels]
