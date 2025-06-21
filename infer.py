#!/usr/bin/env python3
"""Process an image with the trained neural network
Usage:
    infer.py [options] <yaml-config> <checkpoint> <outputs_root>
    infer.py (-h | --help )

Arguments:
   <yaml-config>                 Path to the yaml hyper-parameter file
   <checkpoint>                  Path to the checkpoint
   <outputs_root>                Root dir to the output

Options:
   -h --help                     Show this screen.
   -d --devices <devices>        Comma seperated GPU devices [default: 0]
   -i --identifier <identifier>    Folder identifier [default: default-identifier]
"""

import datetime
import glob
import os
import os.path as osp
import platform
import pprint
import random
import shlex
import shutil
import signal
import subprocess
import sys
import threading

import numpy as np
import torch
import torch.nn as nn
import yaml
from docopt import docopt
import lscorer
from lscorer.config import configs

from datasets.dyrlayout import collate_infer, DyrLayoutDatasetInfer

from lscorer.models.LineScorer import LineScorer
from lscorer.models.hourglass_pose import hg
from lscorer.models.multitask_learner import MultitaskHead

from utils.post_processing import aggregate_split_image_lines

import numpy as np

def setup_device(args):
    device_name = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = args["--devices"]

    if torch.cuda.is_available():
        device_name = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)
        print("Let's use", len(args["--devices"].split(',')), "GPU(s)!")
    else:
        print("CUDA is not available")
    device = torch.device(device_name)

    return device

def get_data_loader(configs, mode='all'):
    kwargs = {
        "collate_fn": collate_infer,
        "num_workers": configs.io.num_workers if os.name != "nt" else 0,
        "pin_memory": True,
    }

    split_file = configs.io.split_file if mode=='all' else configs.io.split_file

    dataset = DyrLayoutDatasetInfer(configs.io.root_images,
                                    configs.io.root_lcnn_pred_lines,
                                    configs.io.root_layout_heatmap,
                                    split_file)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False if mode=='all' else False,
        batch_size=1,
        **kwargs,
    )

    return data_loader

def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"] or "config/ARkit.yaml"
    configs.update(configs.from_yaml(filename = config_file))

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    device = setup_device(args)
    data_loader = get_data_loader(configs, 'infer')
    epoch_size = len(data_loader)

    checkpoint = torch.load(args["<checkpoint>"], map_location=device)

    hg_pose = hg(
            depth=configs.model.depth,
            head=MultitaskHead,
            num_stacks=configs.model.num_stacks,
            num_blocks=configs.model.num_blocks,
            num_classes=sum(sum(configs.model.head_size, [])),
        )
    model = LineScorer(configs, hg_pose)
    model = nn.DataParallel(model, [int(i) for i in args["--devices"].split(',')])
    model = model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    outdir = args["<outputs_root>"]
    root_results_before_postprocessing = os.path.join(outdir, 'results_before_postprocessing')
    os.makedirs(root_results_before_postprocessing, exist_ok=True)
    try:
        inferrer = lscorer.inferrer.Inferrer(
            device=device,
            model=model,
            data_loader=data_loader,
            out=root_results_before_postprocessing,
            configs = configs
        )
        inferrer.infer()

        with open(configs.io.split_file, 'r') as f:
            file_lines = f.readlines()

        all_ids = [line.strip('\n') for line in file_lines]

        root_final_output = os.path.join(outdir, 'line_scorer_final_results')
        os.makedirs(root_final_output, exist_ok=True)
        aggregate_split_image_lines(all_ids, configs.io.root_split_offset, root_results_before_postprocessing, configs.io.root_images, root_final_output)

    except BaseException:
        # if len(glob.glob(f"{outdir}/viz/*")) <= 1:
        #     shutil.rmtree(outdir)
        # raise
        return

if __name__ == "__main__":
    main()