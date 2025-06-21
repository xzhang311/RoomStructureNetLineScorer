import glob
import json
import math
import os
import random

import numpy as np
import numpy.linalg as LA
import torch
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from PIL import Image
import pickle
import cv2
import random

np.random.seed(1024)

LINE_AUGMENT_RANGE = 1.5

class DyrLayoutDataset(Dataset):
    def __init__(self, root_images, root_layout_lines, root_line_scores, root_layout_heatmap, split_file, nlines = 64, mode = 'train', image_size = 128):
        self.nlines = nlines

        # split: (train / val)
        with open(split_file, 'r') as f:
            lines = f.readlines()

        self.root_images = root_images
        self.ids = [line.strip('\n') for line in lines]
        self.line_scores_filelist = []
        self.layout_heatmap_filelist = []
        self.image_filelist = []
        self.layout_lines_filelist = []
        self.mode = mode


        for id in self.ids:
            self.image_filelist.append(os.path.join(root_images, id+'.png'))
            self.line_scores_filelist.append(os.path.join(root_line_scores, id+'.pkl'))
            self.layout_heatmap_filelist.append(os.path.join(root_layout_heatmap,id+'.pkl'))
            self.layout_lines_filelist.append(os.path.join(root_layout_lines, id+'.pkl'))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image = Image.open(self.image_filelist[idx])
        image = image.resize((512, 512))
        image = np.asarray(image)
        image = np.transpose(image, [2, 0, 1])

        with open(self.line_scores_filelist[idx], 'rb') as f:
            line_scores = pickle.load(f)

            # load lcnn prediction
        with open(self.layout_lines_filelist[idx], 'rb') as f:
            layout_lines = pickle.load(f)

        line_feature_map = line_scores['feature'][0] # 256x128x128
        lines = line_scores['lines'] # all points are within [0, 480) range.
        groundtruth = line_scores['groundtruth'] # all measure are within [0, 480) range.

        # trun groundtruth to dict of list
        groundtruth = {k: [dic[k] for dic in groundtruth] for k in groundtruth[0]}
        groundtruth['dist'] = np.asarray(groundtruth['dist'])
        groundtruth['trans_vec_p1'] = np.asarray(groundtruth['trans_vec_p1'])
        groundtruth['trans_vec_p2'] = np.asarray(groundtruth['trans_vec_p2'])
        groundtruth['is_layout_line'] = np.asarray(groundtruth['is_layout_line'])
        groundtruth['layout_line_type'] = np.asarray(groundtruth['layout_line_type'])

        if self.mode == 'train':
            lines, groundtruth = self.sample_lines_from_layout(layout_lines, lines, groundtruth)

        if self.mode == 'debug':
            self.paint_sampled_lines(layout_lines, image)

        # wg = 0, ww = 1, wc = 2.
        type = np.zeros(len(groundtruth['is_layout_line'])) + 3
        idx_wg = np.where(groundtruth['layout_line_type']=='wg')
        idx_ww = np.where(groundtruth['layout_line_type']=='ww')
        idx_wc = np.where(groundtruth['layout_line_type']=='wc')

        idx_not_layout = np.where(groundtruth['is_layout_line']==False)

        type[idx_wg] = 0
        type[idx_ww] = 1
        type[idx_wc] = 2
        type[idx_not_layout] = 3


        # turn it to one hot representation
        # groundtruth['layout_line_type'] = self.one_hot_embedding(type, 4)

        groundtruth['layout_line_type'] = type

        with open(self.layout_heatmap_filelist[idx], 'rb') as f:
            layout_heatmap = pickle.load(f)

        # get heatmap
        feat_4_softmax = layout_heatmap['x_4_softmax']
        heatmap = feat_4_softmax[:, 1:, :, :]
        heatmap = np.max(heatmap, axis=1).squeeze()
        # the input outputs are results of pytorch LogSoftmax
        heatmap = np.exp(heatmap) # heatmap 60x60

        # scale every thing to be within [0, 128) value range.
        lines = lines * 128.0/480.0
        heatmap = np.array(Image.fromarray(heatmap).resize((128, 128), Image.BILINEAR))
        heatmap = heatmap[None, :]

        groundtruth['dist'] = groundtruth['dist'] * 128.0/480.0
        groundtruth['trans_vec_p1']  = groundtruth['trans_vec_p1'] * 128.0/480.0
        groundtruth['trans_vec_p2']  = groundtruth['trans_vec_p2'] * 128.0/480.0

        labels = self.sampling_lines(groundtruth['is_layout_line'], groundtruth['layout_line_type'], self.nlines)

        lines = lines[labels]
        trans_vec_p1 = groundtruth['trans_vec_p1'][labels]
        trans_vec_p2 = groundtruth['trans_vec_p2'][labels]
        layout_line_type = groundtruth['layout_line_type'][labels]

        if self.mode == 'train':
            lines, trans_vec_p1, trans_vec_p2 = self.argument_lines(lines, trans_vec_p1, trans_vec_p2)


        return self.ids[idx],\
               image.astype(np.float32), \
               lines.astype(np.float32), \
               [trans_vec_p1.astype(np.float32), trans_vec_p2.astype(np.float32)], \
               layout_line_type.astype(np.long), \
               heatmap.astype(np.float32)

    def paint_sampled_lines(self, layout_lines, image, nsamples = 8):

        image = np.transpose(image, (1, 2, 0))

        for layout_line in layout_lines:
            p1 = np.asarray(layout_line['p1'])
            p2 = np.asarray(layout_line['p2'])
            type = layout_line['type']

            # reorder the points, so it is from left to right
            # and from top to down
            if p1[0] > p2[0]:
                p1, p2 = p2, p1

            if p1[0] == p2[0]:
                if p1[1] > p2[1]:
                    p1, p2 = p2, p1

            l = np.sqrt(np.sum((p2 - p1) ** 2))
            steps = np.round(l)

            v1 = p2 - p1
            v1 = v1 / np.linalg.norm(v1)

            cv2.line(image, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 0), 2)

            for i in range(nsamples):
                starting_len = random.randint(0, steps)
                ending_len = random.randint(starting_len, steps)

                step_len = ending_len - starting_len

                _p1 = (p1 + v1 * starting_len).astype(np.int16)
                _p2 = (_p1 + v1 * step_len).astype(np.int16)

                cv2.line(image, (_p1[0], _p1[1]), (_p2[0], _p2[1]), (0, 0, 255), 2)

        cv2.imwrite('/home/ec2-user/Downloads/layout_sampled_lines.png', image)

    def sample_lines_from_layout(self, layout_lines, lines, groundtruth, nsamples = 8):

        for layout_line in layout_lines:
            p1 = np.asarray(layout_line['p1'])
            p2 = np.asarray(layout_line['p2'])
            type = layout_line['type']

            # reorder the points, so it is from left to right
            # and from top to down
            if p1[0] > p2[0]:
                p1, p2 = p2, p1

            if p1[0] == p2[0]:
                if p1[1] > p2[1]:
                    p1, p2 = p2, p1

            l = np.sqrt(np.sum((p2 - p1) ** 2))
            steps = np.round(l)

            v1 = p2 - p1
            v1 = v1 / np.linalg.norm(v1)

            _line = np.asarray((p1, p2))[None, :, :]
            lines = np.concatenate((lines, _line), axis=0)

            groundtruth['dist'] = np.append(groundtruth['dist'], 0)
            groundtruth['trans_vec_p1'] = np.append(groundtruth['trans_vec_p1'], np.asarray((0, 0))[None, :], axis=0)
            groundtruth['trans_vec_p2'] = np.append(groundtruth['trans_vec_p2'], np.asarray((0, 0))[None, :], axis=0)
            groundtruth['is_layout_line'] = np.append(groundtruth['is_layout_line'], True)
            groundtruth['layout_line_type'] = np.append(groundtruth['layout_line_type'], type)

            for i in range(nsamples):
                starting_len = random.randint(0, steps)
                ending_len = random.randint(starting_len, steps)

                step_len = ending_len - starting_len

                _p1 = (p1 + v1 * starting_len).astype(np.int16)
                _p2 = (_p1 + v1 * step_len).astype(np.int16)

                _line = np.asarray((_p1, _p2))[None, :, :]
                lines = np.concatenate((lines, _line), axis = 0)

                groundtruth['dist'] = np.append(groundtruth['dist'], 0)
                groundtruth['trans_vec_p1'] = np.append(groundtruth['trans_vec_p1'], np.asarray((0, 0))[None, :], axis = 0)
                groundtruth['trans_vec_p2'] = np.append(groundtruth['trans_vec_p2'], np.asarray((0, 0))[None, :], axis = 0)
                groundtruth['is_layout_line'] = np.append(groundtruth['is_layout_line'], True)
                groundtruth['layout_line_type'] = np.append(groundtruth['layout_line_type'], type)

            zx = 0

        return lines, groundtruth

    def argument_lines(self, lines, trans_vec_p1, trans_vec_p2):
        offset = np.random.normal(0, LINE_AUGMENT_RANGE, (lines.shape[0], 2, 2))

        lines = lines + offset

        trans_vec_p1 = trans_vec_p1 - offset[:, 0, :]
        trans_vec_p2 = trans_vec_p2 - offset[:, 1, :]

        return lines, trans_vec_p1, trans_vec_p2


    def one_hot_embedding(self, labels, num_classes):
        """Embedding labels to one-hot form.

        Args:
          labels: (LongTensor) class labels, sized [N,].
          num_classes: (int) number of classes.

        Returns:
          (tensor) encoded labels, sized [N, #classes].
        """
        y = np.eye(num_classes)
        return y[labels.astype(np.int16)]

    def decide_sampling_amount(self, line_type, nlines):
        type_size = []
        type_size.append(np.sum(line_type==0))
        type_size.append(np.sum(line_type==1))
        type_size.append(np.sum(line_type==2))
        type_size.append(np.sum(line_type==3))
        type_size =np.asarray(type_size)

        nzeros = np.sum(type_size==0)

        if nzeros==0:
            nsamples = [nlines/4, nlines/4, nlines/4, nlines/4]
        elif nzeros==1:
            nsamples = [nlines/3, nlines/3, nlines/3, nlines/3]
            i = np.where(type_size == 0)
            nsamples[i[0][0]]=0
        elif nzeros==2:
            nsamples = [nlines / 2, nlines / 2, nlines / 2, nlines / 2]
            i = np.where(type_size == 0)
            nsamples[i[0][0]] = 0
            nsamples[i[0][1]] = 0
        elif nzeros==3:
            nsamples = [nlines, nlines, nlines, nlines]
            i = np.where(type_size == 0)
            nsamples[i[0][0]] = 0
            nsamples[i[0][1]] = 0
            nsamples[i[0][2]] = 0

        return nsamples

    # def sampling_lines(self, labels, nlines=64):
    #     yes_layouts = np.where(labels == True)
    #     no_layouts = np.where(labels == False)
    #
    #     if len(yes_layouts[0]) != 0:
    #         yes_idx = np.random.choice(range(len(yes_layouts[0])), int(nlines * 0.75))
    #         no_idx = np.random.choice(range(len(no_layouts[0])), int(nlines - int(nlines * 0.75)))
    #
    #         yes_layouts = yes_layouts[0][yes_idx]
    #         no_layouts = no_layouts[0][no_idx]
    #
    #         labels = np.concatenate((yes_layouts, no_layouts))
    #     else:
    #         no_idx = np.random.choice(range(len(no_layouts[0])), int(nlines))
    #         no_layouts = no_layouts[0][no_idx]
    #
    #         labels = no_layouts
    #     return labels

    def sampling_lines(self, labels, line_type, nlines=64):
        nsamples = self.decide_sampling_amount(line_type, nlines)

        type0 = np.where(line_type==0)
        type1 = np.where(line_type==1)
        type2 = np.where(line_type==2)
        type3 = np.where(line_type==3)

        l0 = np.random.choice(range(len(type0[0])), int(nsamples[0]))
        l1 = np.random.choice(range(len(type1[0])), int(nsamples[1]))
        l2 = np.random.choice(range(len(type2[0])), int(nsamples[2]))
        l3 = np.random.choice(range(len(type3[0])), int(nsamples[3]))

        type0_idx = type0[0][l0] if len(l0)!=0 else []
        type1_idx = type1[0][l1] if len(l1)!=0 else []
        type2_idx = type2[0][l2] if len(l2)!=0 else []
        type3_idx = type3[0][l3] if len(l3)!=0 else []

        labels = np.concatenate((type0_idx, type1_idx, type2_idx, type3_idx))
        labels = labels.astype(np.int16)
        return labels

def collate(batch):
    # ids, lines, dists, [trans_vec_p1, trans_vec_p2], layout_line_type, heatmap
    return (
        default_collate([b[0] for b in batch]),
        default_collate([b[1] for b in batch]),
        default_collate([b[2] for b in batch]),
        default_collate([b[3] for b in batch]),
        default_collate([b[4] for b in batch]),
        default_collate([b[5] for b in batch])
    )

# # Testing the dataloader
# def testing_dataloader():
#     root_images = '/mnt/ebs_xizhn2/Data/DYR/OFFLINE_DATASET/lsun_experiment/images'
#     root_line_scores = '/mnt/ebs_xizhn2/Data/DYR/OFFLINE_DATASET/lsun_experiment/line_scores'
#     root_layout_lines = '/mnt/ebs_xizhn2/Data/DYR/OFFLINE_DATASET/lsun_experiment/labels_lines'
#     root_layout_heatmap = '/mnt/ebs_xizhn2/Data/DYR/OFFLINE_DATASET/lsun_experiment/heatmap_pred/drn_d_105_024_all_ms_features'
#     split_file = '/mnt/ebs_xizhn2/Data/DYR/OFFLINE_DATASET/lsun_experiment/train_ids.txt'
#     dyr_dataset = DyrLayoutDataset(root_images, root_layout_lines, root_line_scores, root_layout_heatmap, split_file, nlines = 288, mode='debug')
#
#     kwargs = {
#         "num_workers": 1,
#         "pin_memory": True,
#         "collate_fn": collate
#     }
#
#     train_loader = torch.utils.data.DataLoader(
#         dyr_dataset,
#         shuffle=True,
#         batch_size=2,
#         **kwargs
#     )
#
#     for idx, (id, image, lines, trans_vecs, layout_line_type, line_feature_map, heatmap) in enumerate(train_loader):
#         print('lines shape: {}'.format(lines.shape))
#         print('image shape: {}'.format(image.shape))
#         print('trans_vecs shape: {}'.format(trans_vecs[0].shape))
#         print('layout_line_type shape: {}'.format(layout_line_type.shape))
#         print('heatmap shape: {}'.format(heatmap.shape))
#         print('Done')
#
# testing_dataloader()