#!/usr/bin/env python3
"""Train L-CNN
Usage:
    train.py [options] <yaml-config>
    train.py (-h | --help )

Arguments:
   <yaml-config>                   Path to the yaml hyper-parameter file

Options:
   -h --help                       Show this screen.
   -d --devices <devices>          Comma seperated GPU devices [default: 0]
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

from datasets.dyrlayout import collate, DyrLayoutDataset

from lscorer.models.LineScorer import LineScorer
from lscorer.models.hourglass_pose import hg
from lscorer.models.multitask_learner import MultitaskHead

def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    ret = subprocess.check_output(shlex.split(cmd)).strip()
    if isinstance(ret, bytes):
        ret = ret.decode()
    return ret

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

def setup_random_seeds():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

def get_data_loader(configs, mode='train'):
    kwargs = {
        "collate_fn": collate,
        "num_workers": configs.io.num_workers if os.name != "nt" else 0,
        "pin_memory": True,
    }

    split_file = configs.io.split_file_train if mode=='train' else configs.io.split_file_val

    dataset = DyrLayoutDataset(configs.io.root_images,
                               configs.io.root_layout_lines,
                               configs.io.root_line_scores,
                               configs.io.root_layout_heatmap,
                               split_file,
                               configs.model.n_lines_per_image,
                               mode)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True if mode=='train' else False,
        batch_size=configs.model.batch_size if mode=='train' else configs.model.batch_size_eval,
        **kwargs,
    )


    return data_loader

def get_outdir(identifier):
    # load config
    name = str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
    # name += "-%s" % git_hash()
    name += "-%s" % identifier
    outdir = osp.join(osp.expanduser(configs.io.logdir), name)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    configs.io.resume_from = outdir
    configs.to_yaml(osp.join(outdir, "config.yaml"))
    os.system(f"git diff HEAD > {outdir}/gitdiff.patch")
    return outdir

def get_optimizer(configs, model):
    if configs.optim.name == "Adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=configs.optim.lr,
            weight_decay=configs.optim.weight_decay,
            amsgrad=configs.optim.amsgrad,
        )
    elif configs.optim.name == "SGD":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=configs.optim.lr,
            weight_decay=configs.optim.weight_decay,
            momentum=configs.optim.momentum,
        )
    else:
        raise NotImplementedError

    return optim

def main():
    args = docopt(__doc__)
    config_file = args["<yaml-config>"] or "config/lsun.yaml"
    configs.update(configs.from_yaml(filename = config_file))

    resume_from = configs.io.resume_from

    device = setup_device(args)

    train_loader = get_data_loader(configs, 'train')
    val_loader = get_data_loader(configs, 'val')
    epoch_size = len(train_loader)

    if resume_from:
        checkpoint = torch.load(os.path.join(configs.io.resume_from, "checkpoint_latest.pth"))

    hg_pose = hg(
            depth=configs.model.depth,
            head=MultitaskHead,
            num_stacks=configs.model.num_stacks,
            num_blocks=configs.model.num_blocks,
            num_classes=sum(sum(configs.model.head_size, [])),
        )
    model = LineScorer(configs, hg_pose)

    model = nn.DataParallel(model, [int(i) for i in args["--devices"].split(',')])

    if resume_from:
        model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    optimizer = get_optimizer(configs, model)

    if resume_from:
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
    outdir = resume_from or get_outdir(args["--identifier"])
    print("outdir:", outdir)

    try:
        trainer = lscorer.trainer.Trainer(
            device=device,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            out=outdir,
            configs = configs
        )
        if resume_from:
            trainer.iteration = checkpoint["iteration"]
            if trainer.iteration % epoch_size != 0:
                print("WARNING: iteration is not a multiple of epoch_size, reset it")
                trainer.iteration -= trainer.iteration % epoch_size
            trainer.best_mean_loss = checkpoint["best_mean_loss"]
            del checkpoint
        trainer.train()
    except BaseException:
        if len(glob.glob(f"{outdir}/viz/*")) <= 1:
            shutil.rmtree(outdir)
        raise

if __name__ == "__main__":
    main()