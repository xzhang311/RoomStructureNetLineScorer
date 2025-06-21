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

import math

import torch.onnx
from PIL import Image
import cv2

from sklearn.metrics import confusion_matrix

class Trainer(object):
    def __init__(self, device, model, optimizer, train_loader, val_loader, out, configs):
        self.device = device

        self.model = model
        self.optim = optimizer

        self.configs = configs

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.batch_size = self.configs.model.batch_size

        self.validation_interval = self.configs.io.validation_interval

        self.out = out

        if not os.path.exists(self.out):
            os.makedirs(self.out)

        self.run_tensorboard()
        time.sleep(1)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = self.configs.optim.max_epoch
        self.lr_decay_epoch = self.configs.optim.lr_decay_epoch
        self.mean_loss = self.best_mean_loss = 1e1000

        self.loss_labels = None
        self.avg_metrics = None
        self.metrics = np.zeros(0)

    def run_tensorboard(self):
        board_out = osp.join(self.out, "tensorboard")
        if not osp.exists(board_out):
            os.makedirs(board_out)
        self.writer = SummaryWriter(board_out)
        self.precision_writer = SummaryWriter(osp.join(board_out, 'precision'))
        self.recall_writer = SummaryWriter(osp.join(board_out, 'recall'))

        self.pred_img_writer = []
        for i in range(configs.model.batch_size_eval):
            self.pred_img_writer.append(SummaryWriter(os.path.join(board_out, 'Imgs', str(i))))

        # os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # p = subprocess.Popen(
        #     ["/home/ec2-user/anaconda3/envs/dl_p37/bin/tensorboard", f"--logdir={board_out}", f"--port={self.configs.io.tensorboard_port}"]
        # )
        #
        # def killme():
        #     os.kill(p.pid, signal.SIGTERM)
        #
        # atexit.register(killme)

    def _write_precision_recall(self,  pre_rec, prefix):
        # wg
        self.precision_writer.add_scalar(f"{prefix}/wg", pre_rec[0], self.iteration)
        self.recall_writer.add_scalar(f"{prefix}/wg", pre_rec[3], self.iteration)

        # ww
        self.precision_writer.add_scalar(f"{prefix}/ww", pre_rec[1], self.iteration)
        self.recall_writer.add_scalar(f"{prefix}/ww", pre_rec[4], self.iteration)

        # wc
        self.precision_writer.add_scalar(f"{prefix}/wc", pre_rec[2], self.iteration)
        self.recall_writer.add_scalar(f"{prefix}/wc", pre_rec[5], self.iteration)

    def _write_metrics(self, size, total_loss, prefix, do_print=False):
        for i, metrics in enumerate(self.metrics):
            for label, metric in zip(self.loss_labels, metrics):
                self.writer.add_scalar(
                    f"{prefix}/{i}/{label}", metric / size, self.iteration
                )
            if i == 0 and do_print:
                csv_str = (
                    f"{self.epoch:03}/{self.iteration * self.batch_size:07},"
                    + ",".join(map("{:.11f}".format, metrics / size))
                )
                prt_str = (
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                    + "| ".join(map("{:.5f}".format, metrics / size))
                )
                with open(f"{self.out}/loss.csv", "a") as fout:
                    print(csv_str, file=fout)
                pprint(prt_str, " " * 7)
        self.writer.add_scalar(
            f"{prefix}/total_loss", total_loss / size, self.iteration
        )

        return total_loss

    def _write_images(self, idx, img, line, pred_line_type, target_line_type, epoch):

        target_img = img.copy()
        pred_img = img.copy()

        color = [[255, 0, 0],
                 [0, 255, 0],
                 [0, 0, 255],
                 [170, 170, 170]]

        line = line.data.cpu().numpy()

        for i in range(line.shape[0]):
            scale = img.shape[0]*1.0/128

            x1 = (line[i, 0, 0] * scale).astype(np.int16)
            y1 = (line[i, 0, 1] * scale).astype(np.int16)
            x2 = (line[i, 1, 0] * scale).astype(np.int16)
            y2 = (line[i, 1, 1] * scale).astype(np.int16)

            # wg:0, ww:1, wc:2, no_layout:3
            cv2.line(target_img, (x1, y1), (x2, y2), color[target_line_type[i]], 3)
            cv2.line(pred_img, (x1, y1), (x2, y2), color[pred_line_type[i]], 3)

        self.pred_img_writer[idx].add_image('GT', target_img, epoch)
        self.pred_img_writer[idx].add_image('pred', pred_img, epoch)

    def _confusion_matrix(self, pred, target):
        target = target.reshape(target.size(0) * target.size(1))

        target = target.data.cpu().numpy()
        pred = pred.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)

        cm = confusion_matrix(target, pred, labels=[0, 1, 2, 3])

        return cm

    def _loss_classification(self, pred, target):
        target = target.reshape(target.size(0)*target.size(1))
        lfunc = nn.CrossEntropyLoss()
        loss = lfunc(pred, target)

        return loss

    def _precision(self, confusion_matrix, label):
        rows, cols = confusion_matrix.shape[0], confusion_matrix.shape[1]

        precision = []

        for i in range(rows):
            p = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
            if math.isnan(p):
                precision.append(0)
            else:
                precision.append(p)

        return precision[label]

    def _recall(self, confusion_matrix, label):
        rows, cols = confusion_matrix.shape[0], confusion_matrix.shape[1]

        recall = []

        for i in range(cols):
            r = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
            if math.isnan(r):
                recall.append(0)
            else:
                recall.append(r)

        return recall[label]

    def _precision_recall(self, pred, target):
        if self.loss_labels is None:
            self.loss_labels = ["sum"] + ["classification"] + ["p/c (wg)"] + ["p/c (ww)"] + ["p/c (wc)"]
            self.metrics = np.zeros([1, len(self.loss_labels)])

        confusion_matrix = self._confusion_matrix(pred, target)

        pre_wg = self._precision(confusion_matrix, 0)
        rec_wg = self._recall(confusion_matrix, 0)

        pre_ww = self._precision(confusion_matrix, 1)
        rec_ww = self._recall(confusion_matrix, 1)

        pre_wc = self._precision(confusion_matrix, 2)
        rec_wc = self._recall(confusion_matrix, 2)

        return np.asarray([pre_wg, pre_ww, pre_wc, rec_wg, rec_ww, rec_wc])

    def _loss(self, pred, target):
        if self.loss_labels is None:
            self.loss_labels = ["sum"] + ["classification"]
            self.metrics = np.zeros([1, len(self.loss_labels)])

            print()
            print(
                "| ".join(
                    ["progress "]
                    + list(map("{:7}".format, self.loss_labels))
                    + ["speed"]
                )
            )

            with open(f"{self.out}/loss.csv", "a") as fout:
                print(",".join(["progress"] + self.loss_labels), file=fout)

        total_loss = 0
        for j, name in enumerate(self.loss_labels):
            if name == 'sum':
                continue
            if name == 'classification':
                loss_classification = self._loss_classification(pred, target)
            loss = loss_classification
            self.metrics[0, 0] += loss.item()
            self.metrics[0, j] += loss.item()
            total_loss += loss
        return total_loss

    def validate(self):
        tprint("Running validation...", " " * 75)
        training = self.model.training
        self.model.eval()

        viz = osp.join(self.out, "viz", f"{self.iteration * configs.model.batch_size_eval:09d}")
        npz = osp.join(self.out, "npz", f"{self.iteration * configs.model.batch_size_eval:09d}")
        osp.exists(viz) or os.makedirs(viz)
        osp.exists(npz) or os.makedirs(npz)

        total_loss = 0
        self.metrics[...] = 0
        pre_and_rec = 0

        with torch.no_grad():
            for batch_idx, (ids, images, lines, trans_vecs, layout_lines_types, heatmap) in enumerate(
                    self.val_loader):
                images = recursive_to(images, self.device)
                lines = recursive_to(lines, self.device)
                layout_lines_types = recursive_to(layout_lines_types, self.device)
                trans_vecs = recursive_to(trans_vecs, self.device)
                heatmap = recursive_to(heatmap, self.device)

                ys, y_probs = self.model(images, lines, trans_vecs, heatmap)
                total_loss += self._loss(y_probs, layout_lines_types)
                pre_and_rec += self._precision_recall(y_probs, layout_lines_types)

                if batch_idx==0:
                    y_probs = y_probs.reshape(configs.model.batch_size_eval, configs.model.n_lines_per_image, 4)
                    for i in range(configs.model.batch_size_eval):
                        id = ids[i]
                        img = Image.open(os.path.join(configs.io.root_images, id+'.png'))
                        img = np.asarray(img)
                        y_prob = (y_probs[i]).data.cpu().numpy()
                        pred_line_type = np.argmax(y_prob, axis = 1)
                        target_line_type = layout_lines_types[i]

                        self._write_images(i, img, lines[i], pred_line_type, target_line_type, self.epoch)

                    zx = 0

        self._write_metrics(len(self.val_loader), total_loss, "validation", True)
        self._write_precision_recall(pre_and_rec/len(self.val_loader), 'validation')

        self.mean_loss = total_loss / len(self.val_loader)

        torch.save(
            {
                "iteration": self.iteration,
                "arch": self.model.__class__.__name__,
                "optim_state_dict": self.optim.state_dict(),
                "model_state_dict": self.model.state_dict(),
                "best_mean_loss": self.best_mean_loss,
            },
            osp.join(self.out, "checkpoint_latest.pth"),
        )
        shutil.copy(
            osp.join(self.out, "checkpoint_latest.pth"),
            osp.join(npz, "checkpoint.pth"),
        )
        if self.mean_loss < self.best_mean_loss:
            self.best_mean_loss = self.mean_loss
            shutil.copy(
                osp.join(self.out, "checkpoint_latest.pth"),
                osp.join(self.out, "checkpoint_best.pth"),
            )

        if training:
            self.model.train()

        return 0

    def train_epoch(self):
        self.model.train()

        time = timer()
        for batch_idx, (id, image, lines, trans_vecs, layout_lines_types, heatmap) in enumerate(self.train_loader):
            self.optim.zero_grad()
            self.metrics[...] = 0

            image = recursive_to(image, self.device)
            lines = recursive_to(lines, self.device)
            layout_lines_types = recursive_to(layout_lines_types, self.device)
            trans_vecs = recursive_to(trans_vecs, self.device)
            heatmap = recursive_to(heatmap, self.device)

            y, y_prob = self.model(image, lines, trans_vecs, heatmap)
            loss = self._loss(y_prob, layout_lines_types)
            pre_and_rec = self._precision_recall(y_prob, layout_lines_types)

            if np.isnan(loss.item()):
                raise ValueError("loss is nan while training")
            loss.backward()
            self.optim.step()

            if self.avg_metrics is None:
                self.avg_metrics = self.metrics
            else:
                self.avg_metrics = self.avg_metrics * 0.9 + self.metrics * 0.1

            self._write_metrics(1, loss.item(), "training", do_print=False)

            self._write_precision_recall(pre_and_rec, "training")

            if self.iteration % 4 == 0:
                tprint(
                    f"{self.epoch:03}/{self.iteration * self.batch_size // 1000:04}k| "
                    + "| ".join(map("{:.5f}".format, self.avg_metrics[0]))
                    + f"| {4 * self.batch_size / (timer() - time):04.1f} "
                )
                time = timer()
            # num_images = self.batch_size * self.iteration
            # if num_images % self.validation_interval == 0 or num_images == 600:
            #     self.validate()
            #     time = timer()
            self.iteration += 1

        self.validate()
        return 0

    def train(self):
        epoch_size = len(self.train_loader)
        start_epoch = self.iteration // epoch_size
        for self.epoch in range(start_epoch, self.max_epoch):
            if self.epoch == self.lr_decay_epoch:
                self.optim.param_groups[0]["lr"] /= 10
            self.train_epoch()

def tprint(*args):
    """Temporarily prints things on the screen"""
    print("\r", end="")
    print(*args, end="")

def pprint(*args):
    """Permanently prints things on the screen"""
    print("\r", end="")
    print(*args)