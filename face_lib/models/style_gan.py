import math
import random
import functools
import operator
from collections import abc

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

# from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix

import contextlib
import warnings



# class UpFirDn2dBackward(Function):
#     @staticmethod
#     def forward(
#         ctx, grad_output, kernel, grad_kernel, up, down, pad, g_pad, in_size, out_size
#     ):
#
#         up_x, up_y = up
#         down_x, down_y = down
#         g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1 = g_pad
#
#         grad_output = grad_output.reshape(-1, out_size[0], out_size[1], 1)
#
#         grad_input = upfirdn2d_op.upfirdn2d(
#             grad_output,
#             grad_kernel,
#             down_x,
#             down_y,
#             up_x,
#             up_y,
#             g_pad_x0,
#             g_pad_x1,
#             g_pad_y0,
#             g_pad_y1,
#         )
#         grad_input = grad_input.view(in_size[0], in_size[1], in_size[2], in_size[3])
#
#         ctx.save_for_backward(kernel)
#
#         pad_x0, pad_x1, pad_y0, pad_y1 = pad
#
#         ctx.up_x = up_x
#         ctx.up_y = up_y
#         ctx.down_x = down_x
#         ctx.down_y = down_y
#         ctx.pad_x0 = pad_x0
#         ctx.pad_x1 = pad_x1
#         ctx.pad_y0 = pad_y0
#         ctx.pad_y1 = pad_y1
#         ctx.in_size = in_size
#         ctx.out_size = out_size
#
#         return grad_input
#
#     @staticmethod
#     def backward(ctx, gradgrad_input):
#         kernel, = ctx.saved_tensors
#
#         gradgrad_input = gradgrad_input.reshape(-1, ctx.in_size[2], ctx.in_size[3], 1)
#
#         gradgrad_out = upfirdn2d_op.upfirdn2d(
#             gradgrad_input,
#             kernel,
#             ctx.up_x,
#             ctx.up_y,
#             ctx.down_x,
#             ctx.down_y,
#             ctx.pad_x0,
#             ctx.pad_x1,
#             ctx.pad_y0,
#             ctx.pad_y1,
#         )
#         # gradgrad_out = gradgrad_out.view(ctx.in_size[0], ctx.out_size[0], ctx.out_size[1], ctx.in_size[3])
#         gradgrad_out = gradgrad_out.view(
#             ctx.in_size[0], ctx.in_size[1], ctx.out_size[0], ctx.out_size[1]
#         )
#
#         return gradgrad_out, None, None, None, None, None, None, None, None



# class UpFirDn2d(Function):
#     @staticmethod
#     def forward(ctx, input, kernel, up, down, pad):
#         up_x, up_y = up
#         down_x, down_y = down
#         pad_x0, pad_x1, pad_y0, pad_y1 = pad
#
#         kernel_h, kernel_w = kernel.shape
#         batch, channel, in_h, in_w = input.shape
#         ctx.in_size = input.shape
#
#         input = input.reshape(-1, in_h, in_w, 1)
#
#         ctx.save_for_backward(kernel, torch.flip(kernel, [0, 1]))
#
#         out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
#         out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x
#         ctx.out_size = (out_h, out_w)
#
#         ctx.up = (up_x, up_y)
#         ctx.down = (down_x, down_y)
#         ctx.pad = (pad_x0, pad_x1, pad_y0, pad_y1)
#
#         g_pad_x0 = kernel_w - pad_x0 - 1
#         g_pad_y0 = kernel_h - pad_y0 - 1
#         g_pad_x1 = in_w * up_x - out_w * down_x + pad_x0 - up_x + 1
#         g_pad_y1 = in_h * up_y - out_h * down_y + pad_y0 - up_y + 1
#
#         ctx.g_pad = (g_pad_x0, g_pad_x1, g_pad_y0, g_pad_y1)
#
#         out = upfirdn2d_op.upfirdn2d(
#             input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
#         )
#         # out = out.view(major, out_h, out_w, minor)
#         out = out.view(-1, channel, out_h, out_w)
#
#         return out
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         kernel, grad_kernel = ctx.saved_tensors
#
#         grad_input = None
#
#         if ctx.needs_input_grad[0]:
#             grad_input = UpFirDn2dBackward.apply(
#                 grad_output,
#                 kernel,
#                 grad_kernel,
#                 ctx.up,
#                 ctx.down,
#                 ctx.pad,
#                 ctx.g_pad,
#                 ctx.in_size,
#                 ctx.out_size,
#             )
#
#         return grad_input, None, None, None, None



def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    if not isinstance(up, abc.Iterable):
        up = (up, up)

    if not isinstance(down, abc.Iterable):
        down = (down, down)

    if len(pad) == 2:
        pad = (pad[0], pad[1], pad[0], pad[1])

    # if input.device.type == "cpu":
    #     out = upfirdn2d_native(input, kernel, *up, *down, *pad)
    #
    # else:
    #     out = UpFirDn2d.apply(input, kernel, up, down, pad)

    out = upfirdn2d_native(input, kernel, *up, *down, *pad)

    return out


def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h + down_y) // down_y
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w + down_x) // down_x

    return out.view(-1, channel, out_h, out_w)


# ======================================================================================================================

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # if could_use_op(input):
    #     return conv2d_gradfix(
    #         transpose=False,
    #         weight_shape=weight.shape,
    #         stride=stride,
    #         padding=padding,
    #         output_padding=0,
    #         dilation=dilation,
    #         groups=groups,
    #     ).apply(input, weight, bias)

    # The mistake might be here
    return F.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )



class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))

        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=0.2
            )
            * scale
        )

    else:
        return F.leaky_relu(input, negative_slope=0.2) * scale


# ======================================================================================================================


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        # out = conv2d_gradfix.conv2d(
        out = conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out