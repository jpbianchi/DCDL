"""This flow will train a neural network to perform sentiment classification 
for the beauty products reviews.
"""

import os
import torch
import random
import numpy as np
import pandas as pd
from os.path import join
from pathlib import Path
from pprint import pprint
from torch.utils.data import DataLoader, TensorDataset

from metaflow import FlowSpec, step, Parameter
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from cleanlab.filter import find_label_issues
from sklearn.model_selection import KFold

from src.system import ReviewDataModule, SentimentClassifierSystem
from src.utils import load_config, to_json
from src.consts import DATA_DIR


class TrainBaseline(FlowSpec):
  r"""A MetaFlow that trains a sentiment classifier on reviews of luxury beauty
  products using PyTorch Lightning, identifies data quality issues using CleanLab, 
  and prepares them for review in LabelStudio.

  Arguments
  ---------
  config (str, default: ./config.py): path to a configuration file
  """
  config_path = Parameter('config', 
    help = 'path to config file', default='./config.json')

  @step
  def start(self):
    r"""Start node.
    Set random seeds for reproducibility.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    self.next(self.init_system)

  @step
  def init_system(self):
    r"""Instantiates a data module, pytorch lightning module, 
    and lightning trainer instance.
    """
    # configuration files contain all hyperparameters
    config = load_config(self.config_path)

    # a data module wraps around training, dev, and test datasets
    dm = ReviewDataModule(config)

    # a PyTorch Lightning system wraps around model logic
    system = SentimentClassifierSystem(config)

    # a callback to save best model weights
    checkpoint_callback = ModelCheckpoint(
      dirpath = config.train.ckpt_dir,
      monitor = 'dev_loss',
      mode = 'min',    # look for lowest `dev_loss`
      save_top_k = 1,  # save top 1 checkpoints
      verbose = True,
    )

    trainer = Trainer(
      max_epochs = config.train.optimizer.max_epochs,
      callbacks = [checkpoint_callback])

    # when we save these objects to a `step`, they will be available
    # for use in the next step, through not steps after.
    self.dm = dm
    self.system = system
    self.trainer = trainer
    self.config = config

    self.next(self.train_test)

  @step
  def train_test(self):
    """Calls `fit` on the trainer.
    
    We first train and (offline) evaluate the model to see what 
    performance would be without any improvements to data quality.
    """
    # Call `fit` on the trainer with `system` and `dm`.
    # Our solution is one line.
    self.trainer.fit(self.system, self.dm)
    self.trainer.test(self.system, self.dm, ckpt_path = 'best')

    # results are saved into the system
    results = self.system.test_results

    # print results to command line
    pprint(results)

    log_file = join(Path(__file__).resolve().parent.parent, 
      f'logs', 'pre-results.json')

    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save to disk

    self.next(self.end)

  @step
  def end(self):
    """End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python flow_baseline.py`. To list
  this flow, run `python flow_baseline.py show`. To execute
  this flow, run `python flow_baseline.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python flow_baseline.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python flow_baseline.py resume`
  
  You can specify a run id as well.
  """
  flow = TrainBaseline()
# first, download the big lfs files
# cd data
# sudo git lfs pull (you will have to provideyour github id and pw)


# python flow_baseline.py --no-pylint run
# Metaflow 2.6.0 executing TrainBaseline for user:gitpod
# Validating your flow...
#     The graph looks good!
# 2023-10-06 22:28:26.930 Workflow starting (run-id 1696631306926666):
# 2023-10-06 22:28:26.948 [1696631306926666/start/1 (pid 20805)] Task is starting.
# 2023-10-06 22:28:29.874 [1696631306926666/start/1 (pid 20805)] Task finished successfully.
# 2023-10-06 22:28:29.888 [1696631306926666/init_system/2 (pid 20851)] Task is starting.
# 2023-10-06 22:28:32.447 [1696631306926666/init_system/2 (pid 20851)] Split is train
# 2023-10-06 22:28:32.541 [1696631306926666/init_system/2 (pid 20851)] Split is dev
# 2023-10-06 22:28:32.600 [1696631306926666/init_system/2 (pid 20851)] Split is test
# 2023-10-06 22:28:32.672 [1696631306926666/init_system/2 (pid 20851)] GPU available: False, used: False
# 2023-10-06 22:28:32.672 [1696631306926666/init_system/2 (pid 20851)] TPU available: False, using: 0 TPU cores
# 2023-10-06 22:28:32.672 [1696631306926666/init_system/2 (pid 20851)] IPU available: False, using: 0 IPUs
# 2023-10-06 22:28:32.673 [1696631306926666/init_system/2 (pid 20851)] HPU available: False, using: 0 HPUs
# 2023-10-06 22:28:54.058 [1696631306926666/init_system/2 (pid 20851)] Task finished successfully.
# 2023-10-06 22:28:54.079 [1696631306926666/train_test/3 (pid 20954)] Task is starting.
# 2023-10-06 22:29:01.309 [1696631306926666/train_test/3 (pid 20954)] Missing logger folder: /workspace/DCDL/course/week3/conflearn_project/lightning_logs
# 2023-10-06 22:29:01.311 [1696631306926666/train_test/3 (pid 20954)] 
# Epoch 1:  80%|████████  | 1600/2000 [00:25<00:06, 62.75it/s, loss=0.42, v_num=0, train_loss=0.354, train_acc=0.906, dev_loss=0.564, dev_acc=0.789]]
# 2023-10-06 22:29:56.885 [1696631306926666/train_test/3 (pid 20954)] | Name  | Type       | Params|██████████| 400/400 [00:04<00:00, 105.16it/s]
# 2023-10-06 22:29:56.885 [1696631306926666/train_test/3 (pid 20954)] -------------------------------------
# 2023-10-06 22:29:56.886 [1696631306926666/train_test/3 (pid 20954)] 0 | model | Sequential | 49.3 K400 [00:00<?, ?it/s]
# Epoch 1:  80%|████████  | 1601/2000 [00:25<00:06, 62.63it/s, loss=0.42, v_num=0, train_loss=0.354, train_acc=0.906, dev_loss=0.564, dev_acc=0.789]
# Epoch 1:  80%|████████  | 1602/2000 [00:25<00:06, 62.66it/s, loss=0.42, v_num=0, train_loss=0.354, train_acc=0.906, dev_loss=0.564, dev_acc=0.789]
# Epoch 1:  80%|████████  | 1603/2000 [00:25<00:06, 62.68it/s, loss=0.42, v_num=0, train_loss=0.354, train_acc=0.906, dev_loss=0.564, dev_acc=0.789]
# Epoch 1:  80%|████████  | 1604/2000 [00:25<00:06, 62.71it/s, loss=0.42, v_num=0, train_loss=0.354, train_acc=0.906, dev_loss=0.564, dev_acc=0.789]
# Epoch 1:  80%|████████  | 1605/2000 [00:25<00:06, 62.73it/s, loss=0.42, v_num=0, train_loss=0.354, train_acc=0.906, dev_loss=0.564, dev_acc=0.789]
# Epoch 2:  80%|████████  | 1600/2000 [00:24<00:06, 66.44it/s, loss=0.346, v_num=0, train_loss=0.342, train_acc=0.875, dev_loss=0.432, dev_acc=0.830]del to '/workspace/DCDL/course/week3/conflearn_project/log/epoch=0-step=1600.ckpt' as top 1
# Epoch 3:  80%|████████  | 1600/2000 [00:23<00:05, 68.11it/s, loss=0.369, v_num=0, train_loss=0.379, train_acc=0.812, dev_loss=0.373, dev_acc=0.854]del to '/workspace/DCDL/course/week3/conflearn_project/log/epoch=1-step=3200.ckpt' as top 1
# Epoch 4:  80%|████████  | 1600/2000 [00:23<00:05, 68.13it/s, loss=0.316, v_num=0, train_loss=0.247, train_acc=0.938, dev_loss=0.351, dev_acc=0.851]del to '/workspace/DCDL/course/week3/conflearn_project/log/epoch=2-step=4800.ckpt' as top 1
# Epoch 5:  80%|████████  | 1600/2000 [00:26<00:06, 60.34it/s, loss=0.333, v_num=0, train_loss=0.200, train_acc=0.938, dev_loss=0.324, dev_acc=0.869]del to '/workspace/DCDL/course/week3/conflearn_project/log/epoch=3-step=6400.ckpt' as top 1
# Epoch 6:  80%|████████  | 1600/2000 [00:23<00:05, 69.28it/s, loss=0.283, v_num=0, train_loss=0.227, train_acc=0.969, dev_loss=0.320, dev_acc=0.866]del to '/workspace/DCDL/course/week3/conflearn_project/log/epoch=4-step=8000.ckpt' as top 1
# Epoch 7:  80%|████████  | 1600/2000 [00:23<00:05, 69.29it/s, loss=0.322, v_num=0, train_loss=0.265, train_acc=0.875, dev_loss=0.314, dev_acc=0.868]del to '/workspace/DCDL/course/week3/conflearn_project/log/epoch=5-step=9600.ckpt' as top 1
# Epoch 8:  80%|████████  | 1600/2000 [00:23<00:05, 68.17it/s, loss=0.324, v_num=0, train_loss=0.205, train_acc=0.906, dev_loss=0.305, dev_acc=0.878] del to '/workspace/DCDL/course/week3/conflearn_project/log/epoch=6-step=11200.ckpt' as top 1
# Epoch 9:  80%|████████  | 1600/2000 [00:24<00:06, 65.87it/s, loss=0.29, v_num=0, train_loss=0.426, train_acc=0.750, dev_loss=0.290, dev_acc=0.882]] del to '/workspace/DCDL/course/week3/conflearn_project/log/epoch=7-step=12800.ckpt' as top 1
# Epoch 9: 100%|██████████| 2000/2000 [00:28<00:00, 70.64it/s, loss=0.29, v_num=0, train_loss=0.426, train_acc=0.750, dev_loss=0.288, dev_acc=0.884]model to '/workspace/DCDL/course/week3/conflearn_project/log/epoch=8-step=14400.ckpt' as top 1
# 2023-10-06 22:34:15.727 [1696631306926666/train_test/3 (pid 20954)] Epoch 9, global step 16000: 'dev_loss' reached 0.28792 (best 0.28792), saving model to '/workspace/DCDL/course/week3/conflearn_project/log/epoch=9-step=16000.ckpt' as top 1
# 2023-10-06 22:34:15.731 [1696631306926666/train_test/3 (pid 20954)] Restoring states from the checkpoint path at /workspace/DCDL/course/week3/conflearn_project/log/epoch=9-step=16000.ckpt
# 2023-10-06 22:34:15.783 [1696631306926666/train_test/3 (pid 20954)] Loaded model weights from checkpoint at /workspace/DCDL/course/week3/conflearn_project/log/epoch=9-step=16000.ckpt
# Testing DataLoader 0: 100%|██████████| 500/500 [00:05<00:00, 95.58it/s] ing: 0it [00:00, ?it/s]
# 2023-10-06 22:34:21.024 [1696631306926666/train_test/3 (pid 20954)] {'acc': 0.8872500061988831, 'loss': 0.29152730107307434} << FINAL ACC 0.88
# 2023-10-06 22:35:24.446 [1696631306926666/train_test/3 (pid 20954)] Task finished successfully.
# 2023-10-06 22:35:24.463 [1696631306926666/end/4 (pid 21873)] Task is starting.
# 2023-10-06 22:35:26.855 [1696631306926666/end/4 (pid 21873)] done! great work!
# 2023-10-06 22:35:27.276 [1696631306926666/end/4 (pid 21873)] Task finished successfully.
# 2023-10-06 22:35:27.277 Done!s