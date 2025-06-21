import atexit
import os
import os.path as osp
import shutil
import signal
import subprocess
import threading
import time
from timeit import default_timer as timer

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
from tensorboardX import SummaryWriter

from lscorer.config import configs
from utils.utils import recursive_to

import pickle

import math

import torch.onnx
from PIL import Image
import cv2

from sklearn.metrics import confusion_matrix

class Inferrer(object):
    def __init__(self, device, model, data_loader, out, configs):
        self.device = device
        self.model = model
        self.configs = configs
        self.data_loader = data_loader
        self.batch_size = self.configs.model.batch_size
        self.epoch = 0
        self.iteration = 0
        self.out = out

        if not os.path.exists(self.out):
            os.makedirs(self.out)

    def infer(self):
        tprint("Running inference...", " " * 75)

        self.model.eval()

        with torch.no_grad():
            for batch_idx, (ids, image, lines, heatmap, img_w, img_h) in enumerate(
                    self.data_loader):
                print('Processing: {}'.format(ids[0]))

                image = recursive_to(image, self.device)
                lines = recursive_to(lines, self.device)
                heatmap = recursive_to(heatmap, self.device)

                ys, y_probs = self.model(image = image, lines = lines, heatmap = heatmap, trans_vecs = None)
                y_prob = (y_probs).data.cpu().numpy()
                pred_line_type = np.argmax(y_prob, axis=1)

                tmp_image = image.data.cpu().numpy()[0, :]
                tmp_image = np.transpose(tmp_image, (1, 2, 0))

                drawn_image = self.draw_image(ids[0], tmp_image, lines[0, :].data.cpu().numpy(), pred_line_type, img_w.data.cpu().numpy()[0], img_h.data.cpu().numpy()[0])
                out_path = os.path.join(self.out, ids[0] + '.jpg')
                cv2.imwrite(out_path, drawn_image)
                self.save_lines(ids[0], lines[0, :], y_prob, img_w.data.cpu().numpy()[0], img_h.data.cpu().numpy()[0])
                zx = 0

    def save_lines(self, id, lines, prob, img_w, img_h):
        out_path = os.path.join(self.out, id + '.pkl')

        x_scale = img_w * 1.0 / 128
        y_scale = img_h * 1.0 / 128

        lines[:, :, 0] = lines[:, :, 0] * x_scale
        lines[:, :, 1] = lines[:, :, 1] * y_scale

        data = {}
        data['lines'] = lines
        data['prob'] = prob

        with open(out_path, 'wb') as f:
            pickle.dump(data, f)

    def draw_image(self, id, image, lines, pred_line_type, img_w, img_h):
        r = image[:, :, 0]
        b = image[:, :, 2]

        image[:, :, 0] = b
        image[:, :, 2] = r

        color = [[255, 0, 0],
                 [0, 255, 0],
                 [0, 0, 255],
                 [170, 170, 170]]

        x_scale = img_w * 1.0 / 128
        y_scale = img_h * 1.0 / 128

        image = cv2.resize(image, (img_w, img_h), interpolation = cv2.INTER_LANCZOS4)

        for i in range(lines.shape[0]):
            x1 = (lines[i, 0, 0] * x_scale).astype(np.int16)
            y1 = (lines[i, 0, 1] * y_scale).astype(np.int16)
            x2 = (lines[i, 1, 0] * x_scale).astype(np.int16)
            y2 = (lines[i, 1, 1] * y_scale).astype(np.int16)

            # wg:0, ww:1, wc:2, no_layout:3
            cv2.line(image, (x1, y1), (x2, y2), color[pred_line_type[i]], 3)

        return image

def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")

def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)