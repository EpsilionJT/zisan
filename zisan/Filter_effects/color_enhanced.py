#!/usr/bin/env python

'''Converts sequence of images to compact PDF while removing speckles,
bleedthrough, etc.
'''

# for some reason pylint complains about members being undefined :(
# pylint: disable=E1101

from __future__ import print_function

import sys
import os
import re
import subprocess
import shlex

from argparse import ArgumentParser

import numpy as np
from PIL import Image
from scipy.cluster.vq import kmeans, vq
from skimage import io
from matplotlib import pyplot as plt
import cv2
from utils import pngToJpg

np.seterr(divide='ignore',invalid='ignore')

class Options(object):
    def __init__(self):
        self.value_threshold=0.25
        self.sat_threshold=0.2
        self.num_colors=8
        self.sample_fraction=0.05
        self.white_bg=False
        self.saturate=True
        self.sort_numerically=True

######################################################################

def quantize(image, bits_per_channel=None):

    '''Reduces the number of bits per channel in the given image.'''

    if bits_per_channel is None:
        bits_per_channel = 6

    assert image.dtype == np.uint8

    shift = 8-bits_per_channel
    halfbin = (1 << shift) >> 1

    return ((image.astype(int) >> shift) << shift) + halfbin

######################################################################

def pack_rgb(rgb):

    '''Packs a 24-bit RGB triples into a single integer,
    works on both arrays and tuples.'''

    orig_shape = None

    if isinstance(rgb, np.ndarray):
        assert rgb.shape[-1] == 3
        orig_shape = rgb.shape[:-1]
    else:
        assert len(rgb) == 3
        rgb = np.array(rgb)

    rgb = rgb.astype(int).reshape((-1, 3))

    packed = (rgb[:, 0] << 16 |
              rgb[:, 1] << 8 |
              rgb[:, 2])

    if orig_shape is None:
        return packed
    else:
        return packed.reshape(orig_shape)

######################################################################

def unpack_rgb(packed):

    '''Unpacks a single integer or array of integers into one or more
 24-bit RGB values.
    '''

    orig_shape = None

    if isinstance(packed, np.ndarray):
        assert packed.dtype == int
        orig_shape = packed.shape
        packed = packed.reshape((-1, 1))

    rgb = ((packed >> 16) & 0xff,
           (packed >> 8) & 0xff,
           (packed) & 0xff)

    if orig_shape is None:
        return rgb
    else:
        return np.hstack(rgb).reshape(orig_shape + (3,))

######################################################################

def get_bg_color(image, bits_per_channel=None):

    '''
    从图像或RGB颜色数组中获取背景颜色，方法是将相似的颜色分组到存储箱中并找到最常用的颜色。
    '''

    assert image.shape[-1] == 3

    quantized = quantize(image, bits_per_channel).astype(int)
    packed = pack_rgb(quantized)

    unique, counts = np.unique(packed, return_counts=True)

    packed_mode = unique[counts.argmax()]

    return unpack_rgb(packed_mode)

######################################################################

def rgb_to_sv(rgb):

    '''将RGB图像或RGB颜色数组转换为饱和度和r值，并将每个值作为单独的32位浮点数组或r值返回。
    '''

    if not isinstance(rgb, np.ndarray):
        rgb = np.array(rgb)

    axis = len(rgb.shape)-1
    cmax = rgb.max(axis=axis).astype(np.float32)
    cmin = rgb.min(axis=axis).astype(np.float32)
    delta = cmax - cmin

    saturation = delta.astype(np.float32) / cmax.astype(np.float32)
    saturation = np.where(cmax == 0, 0, saturation)

    value = cmax/255.0

    return saturation, value

######################################################################

def postprocess(output_filename, options):

    '''对提供的文件运行后处理命令。'''

    assert options.postprocess_cmd

    base, _ = os.path.splitext(output_filename)
    post_filename = base + options.postprocess_ext

    cmd = options.postprocess_cmd
    cmd = cmd.replace('%i', output_filename)
    cmd = cmd.replace('%o', post_filename)
    cmd = cmd.replace('%e', options.postprocess_ext)

    subprocess_args = shlex.split(cmd)

    if os.path.exists(post_filename):
        os.unlink(post_filename)

    if not options.quiet:
        print('  running "{}"...'.format(cmd), end=' ')
        sys.stdout.flush()

    try:
        result = subprocess.call(subprocess_args)
        before = os.stat(output_filename).st_size
        after = os.stat(post_filename).st_size
    except OSError:
        result = -1

    if result == 0:

        if not options.quiet:
            print('{:.1f}% reduction'.format(
                100*(1.0-float(after)/before)))

        return post_filename

    else:

        sys.stderr.write('warning: postprocessing failed!\n')
        return None

######################################################################

def sample_pixels(img, options):

    '''Pick a fixed percentage of pixels in the image, returned in random
 order.'''

    pixels = img.reshape((-1, 3))
    num_pixels = pixels.shape[0]
    num_samples = int(num_pixels*options.sample_fraction)

    idx = np.arange(num_pixels)
    np.random.shuffle(idx)

    return pixels[idx[:num_samples]]

######################################################################

def get_fg_mask(bg_color, samples, options):

    '''Determine whether each pixel in a set of samples is foreground by
    comparing it to the background color. A pixel is classified as a
    foreground pixel if either its value or saturation differs from the
    background by a threshold.'''

    s_bg, v_bg = rgb_to_sv(bg_color)
    s_samples, v_samples = rgb_to_sv(samples)

    s_diff = np.abs(s_bg - s_samples)
    v_diff = np.abs(v_bg - v_samples)

    return ((v_diff >= options.value_threshold) |
            (s_diff >= options.sat_threshold))

######################################################################

def get_palette(samples, options, return_mask=False, kmeans_iter=40):

    '''提取采样RGB值集的调色板。第一个调色板条目始终是背景色；其余的是通过运行Kmeans聚类从前景像素确定的。返回r调色板以及对应于前景像素的遮罩。
        '''

    bg_color = get_bg_color(samples, 6)

    fg_mask = get_fg_mask(bg_color, samples, options)

    centers, _ = kmeans(samples[fg_mask].astype(np.float32),
                        options.num_colors-1,
                        iter=kmeans_iter)

    palette = np.vstack((bg_color, centers)).astype(np.uint8)

    if not return_mask:
        return palette
    else:
        return palette, fg_mask

######################################################################

def apply_palette(img, palette, options):

    '''
    将托盘应用于给定图像。第一步是将所有背景像素设置为背景色；然后，使用最近邻匹配将每个前景色映射到调色板中最近的一个。
    '''


    bg_color = palette[0]

    fg_mask = get_fg_mask(bg_color, img, options)

    orig_shape = img.shape

    pixels = img.reshape((-1, 3))
    fg_mask = fg_mask.flatten()

    num_pixels = pixels.shape[0]

    labels = np.zeros(num_pixels, dtype=np.uint8)

    labels[fg_mask], _ = vq(pixels[fg_mask], palette)

    return labels.reshape(orig_shape[:-1])

######################################################################

def deal(labels, palette, options):

    '''
    将标签调色板对另存为索引PNG图像。这r通过将最小颜色分量映射到零，将最大颜色分量映射到255，选择性地使托盘饱和，还可以将背景颜色设置为纯白色。
    '''
    if options.saturate:
        palette = palette.astype(np.float32)
        pmin = palette.min()
        pmax = palette.max()
        palette = 255 * (palette - pmin)/(pmax-pmin)
        palette = palette.astype(np.uint8)

    if options.white_bg:
        palette = palette.copy()
        palette[0] = (255, 255, 255)
    output_img = Image.fromarray(labels,'P')
    output_img.putpalette(palette.flatten())
    return output_img

######################################################################

def get_global_palette(filenames, options):

    '''Fetch the global palette for a series of input files by merging
    their samples together into one large array.
    '''

    input_filenames = []

    all_samples = []

    if not options.quiet:
        print('building global palette...')

    for input_filename in filenames:

        img, _ = load(input_filename)
        if img is None:
            continue

        if not options.quiet:
            print('  processing {}...'.format(input_filename))

        samples = sample_pixels(img, options)
        input_filenames.append(input_filename)
        all_samples.append(samples)

    num_inputs = len(input_filenames)

    all_samples = [s[:int(round(float(s.shape[0])/num_inputs))]
                   for s in all_samples]

    all_samples = np.vstack(tuple(all_samples))

    global_palette = get_palette(all_samples, options)

    if not options.quiet:
        print('  done\n')

    return input_filenames, global_palette


def color_enhanced_filter(img):
    options=Options()
    samples = sample_pixels(img, options)
    palette = get_palette(samples, options)
    labels = apply_palette(img, palette, options)
    re=deal(labels, palette, options)
    re.save('FRERTRT.png')
    re=cv2.imread('FRERTRT.png')
    re=pngToJpg(re)
    re=cv2.cvtColor(re,cv2.COLOR_BGR2RGB)
    return re

