"""This flow will train a neural network to perform sentiment classification 
for Amazon reviews across several product categories.
"""

import os
import torch
import random
import numpy as np
from os.path import join
from pprint import pprint

from metaflow import FlowSpec, step, Parameter

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.systems import ReviewDataModule, SentimentClassifierSystem
from src.paths import LOG_DIR, CONFIG_DIR
from src.utils import load_config, to_json


class TrainClassifier(FlowSpec):
  r"""A flow that trains a natural language inference model.

  Arguments
  ---------
  config (str, default: ./configs/train.json): path to a configuration file
  """
  config_path = Parameter('config', 
    help = 'path to config file', default = join(CONFIG_DIR, 'train.json'))

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
    config = load_config(self.config_path) # <<< loads the json config file here

    # a data module wraps around training, dev, and test datasets
    dm = ReviewDataModule(config)

    # a PyTorch Lightning system wraps around model logic
    system = SentimentClassifierSystem(config)

    # a callback to save best model weights
    checkpoint_callback = ModelCheckpoint(
      dirpath = config.system.save_dir,
      monitor = 'dev_loss',
      mode = 'min',    # look for lowest `dev_loss`
      save_top_k = 1,  # save top 1 checkpoints
      verbose = True,
    )

    trainer = Trainer(
      logger = TensorBoardLogger(save_dir=LOG_DIR),
      max_epochs = config.system.optimizer.max_epochs,
      callbacks = [checkpoint_callback])

    # when we save these objects to a `step`, they will be available
    # for use in the next step, through not steps after.
    self.dm = dm
    self.system = system
    self.trainer = trainer

    self.next(self.train_model)

  @step
  def train_model(self):
    """Calls `fit` on the trainer."""

    # Call `fit` on the trainer with `system` and `dm`.
    # Our solution is one line.
    self.trainer.fit(self.system, self.dm)

    self.next(self.offline_test)

  @step
  def offline_test(self):
    r"""Calls (offline) `test` on the trainer. Saves results to a log file."""

    # Load the best checkpoint and compute results using `self.trainer.test`
    self.trainer.test(self.system, self.dm, ckpt_path = 'best')
    results = self.system.test_results

    # print results to command line
    pprint(results)

    log_file = join(LOG_DIR, 'train_flow', 'results.json')
    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save results to disk

    self.next(self.end)

  @step
  def end(self):
    """End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python train_flow.py`. To list
  this flow, run `python train_flow.py show`. To execute
  this flow, run `python train_flow.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python train_flow.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python train_flow.py resume`
  
  You can specify a run id as well.
  """
  flow = TrainClassifier()

  # python train_flow.py run --config configs/update.json

#   Metaflow 2.6.0 executing TrainClassifier for user:gitpod
# Validating your flow...
#     The graph looks good!
# Running pylint...
#     Pylint is happy!
# 2023-10-11 20:01:47.143 Workflow starting (run-id 1697054507139036):
# 2023-10-11 20:01:47.166 [1697054507139036/start/1 (pid 60187)] Task is starting.
# 2023-10-11 20:01:49.643 [1697054507139036/start/1 (pid 60187)] Task finished successfully.
# 2023-10-11 20:01:49.666 [1697054507139036/init_system/2 (pid 60215)] Task is starting.
# 2023-10-11 20:01:53.836 [1697054507139036/init_system/2 (pid 60215)] GPU available: False, used: False
# 2023-10-11 20:01:53.836 [1697054507139036/init_system/2 (pid 60215)] TPU available: False, using: 0 TPU cores
# 2023-10-11 20:01:53.836 [1697054507139036/init_system/2 (pid 60215)] IPU available: False, using: 0 IPUs
# 2023-10-11 20:02:40.052 [1697054507139036/init_system/2 (pid 60215)] HPU available: False, using: 0 HPUs
# 2023-10-11 20:02:40.054 [1697054507139036/init_system/2 (pid 60215)] Task finished successfully.
# 2023-10-11 20:02:40.078 [1697054507139036/train_model/3 (pid 60395)] Task is starting.
# 2023-10-11 20:02:52.601 [1697054507139036/train_model/3 (pid 60395)] 
# Epoch 1:  98%|█████████▊| 1250/1282 [00:31<00:00, 39.15it/s, loss=0.392, v_num=60, train_loss=0.373, train_acc=0.867, dev_loss=0.471, dev_acc=0.816]]]
# 2023-10-11 20:03:57.345 [1697054507139036/train_model/3 (pid 60395)] | Name  | Type       | Params|██████████| 32/32 [00:00<00:00, 62.59it/s]
# 2023-10-11 20:03:57.642 [1697054507139036/train_model/3 (pid 60395)] -------------------------------------
# 2023-10-11 20:03:57.643 [1697054507139036/train_model/3 (pid 60395)] 0 | model | Sequential | 49.3 K32 [00:00<?, ?it/s]
# 2023-10-11 20:03:57.643 [1697054507139036/train_model/3 (pid 60395)] -------------------------------------
# 2023-10-11 20:03:57.643 [1697054507139036/train_model/3 (pid 60395)] 49.3 K    Trainable params
# Epoch 1:  98%|█████████▊| 1251/1282 [00:32<00:00, 38.69it/s, loss=0.392, v_num=60, train_loss=0.373, train_acc=0.867, dev_loss=0.471, dev_acc=0.816]
# Epoch 1:  98%|█████████▊| 1252/1282 [00:32<00:00, 38.71it/s, loss=0.392, v_num=60, train_loss=0.373, train_acc=0.867, dev_loss=0.471, dev_acc=0.816]
# Epoch 1:  98%|█████████▊| 1253/1282 [00:32<00:00, 38.64it/s, loss=0.392, v_num=60, train_loss=0.373, train_acc=0.867, dev_loss=0.471, dev_acc=0.816]
# 2023-10-11 20:03:57.766 [1697054507139036/train_model/3 (pid 60395)] Epoch 0, global step 1250: 'dev_loss' reached 0.47146 (best 0.47146), saving model to '/workspace/DCDL/course/week4/monitor_Epoch 2:  98%|█████████▊| 1250/1282 [00:31<00:00, 40.16it/s, loss=0.359, v_num=60, train_loss=0.364, train_acc=0.859, dev_loss=0.387, dev_acc=0.829]]]
# 2023-10-11 20:04:29.435 [1697054507139036/train_model/3 (pid 60395)] Epoch 1, global step 2500: 'dev_loss' reached 0.38700 (best 0.38700), saving model to '/workspace/DCDL/course/week4/monitor_Epoch 3:  98%|█████████▊| 1250/1282 [00:35<00:00, 34.85it/s, loss=0.339, v_num=60, train_loss=0.324, train_acc=0.883, dev_loss=0.362, dev_acc=0.834]4]
# 2023-10-11 20:05:06.228 [1697054507139036/train_model/3 (pid 60395)] Epoch 2, global step 3750: 'dev_loss' reached 0.36212 (best 0.36212), saving model to '/workspace/DCDL/course/week4/monitor_Epoch 4:  98%|█████████▊| 1250/1282 [00:35<00:00, 35.02it/s, loss=0.348, v_num=60, train_loss=0.341, train_acc=0.828, dev_loss=0.348, dev_acc=0.841]]
# 2023-10-11 20:05:43.929 [1697054507139036/train_model/3 (pid 60395)] Epoch 3, global step 5000: 'dev_loss' reached 0.34837 (best 0.34837), saving model to '/workspace/DCDL/course/week4/monitor_Epoch 5:  98%|█████████▊| 1250/1282 [00:37<00:00, 32.90it/s, loss=0.335, v_num=60, train_loss=0.309, train_acc=0.844, dev_loss=0.330, dev_acc=0.858]]]
# 2023-10-11 20:06:23.768 [1697054507139036/train_model/3 (pid 60395)] Epoch 4, global step 6250: 'dev_loss' reached 0.33006 (best 0.33006), saving model to '/workspace/DCDL/course/week4/monitor_Epoch 6:  98%|█████████▊| 1250/1282 [00:31<00:00, 39.96it/s, loss=0.314, v_num=60, train_loss=0.324, train_acc=0.859, dev_loss=0.329, dev_acc=0.860]0]
# 2023-10-11 20:06:55.952 [1697054507139036/train_model/3 (pid 60395)] Epoch 5, global step 7500: 'dev_loss' reached 0.32894 (best 0.32894), saving model to '/workspace/DCDL/course/week4/monitor_Epoch 7:  98%|█████████▊| 1250/1282 [00:31<00:00, 39.94it/s, loss=0.314, v_num=60, train_loss=0.285, train_acc=0.875, dev_loss=0.341, dev_acc=0.851]]1]
# Epoch 8:  98%|█████████▊| 1250/1282 [00:32<00:00, 38.93it/s, loss=0.322, v_num=60, train_loss=0.419, train_acc=0.828, dev_loss=0.318, dev_acc=0.862]2]
# 2023-10-11 20:08:01.173 [1697054507139036/train_model/3 (pid 60395)] Epoch 7, global step 10000: 'dev_loss' reached 0.31847 (best 0.31847), saving model to '/workspace/DCDL/course/week4/monitorEpoch 9:  98%|█████████▊| 1250/1282 [00:39<00:01, 31.92it/s, loss=0.324, v_num=60, train_loss=0.311, train_acc=0.867, dev_loss=0.318, dev_acc=0.860]]
# 2023-10-11 20:08:41.416 [1697054507139036/train_model/3 (pid 60395)] Epoch 8, global step 11250: 'dev_loss' reached 0.31755 (best 0.31755), saving model to '/workspace/DCDL/course/week4/monitorEpoch 9: 100%|██████████| 1282/1282 [00:40<00:00, 31.67it/s, loss=0.324, v_num=60, train_loss=0.311, train_acc=0.867, dev_loss=0.307, dev_acc=0.867]
# 2023-10-11 20:08:42.630 [1697054507139036/train_model/3 (pid 60395)] Epoch 9, global step 12500: 'dev_loss' reached 0.30709 (best 0.30709), saving model to '/workspace/DCDL/course/week4/monitor_project/artifacts/ckpts/update/epoch=9-step=12500.ckpt' as top 1
# 2023-10-11 20:10:57.927 [1697054507139036/train_model/3 (pid 60395)] Task finished successfully.
# 2023-10-11 20:10:57.955 [1697054507139036/offline_test/4 (pid 63014)] Task is starting.
# 2023-10-11 20:11:29.141 [1697054507139036/offline_test/4 (pid 63014)] Restoring states from the checkpoint path at /workspace/DCDL/course/week4/monitor_project/artifacts/ckpts/update/epoch=9-step=12500.ckpt
# 2023-10-11 20:11:29.151 [1697054507139036/offline_test/4 (pid 63014)] Loaded model weights from checkpoint at /workspace/DCDL/course/week4/monitor_project/artifacts/ckpts/update/epoch=9-step=12500.ckpt
# Testing DataLoader 0: 100%|██████████| 32/32 [00:01<00:00, 27.06it/s] Testing: 0it [00:00, ?it/s]
# 2023-10-11 20:11:31.104 [1697054507139036/offline_test/4 (pid 63014)] {'acc': 0.875, 'loss': 0.311}. # <<< ACCURACY 0.875
# 2023-10-11 20:13:45.615 [1697054507139036/offline_test/4 (pid 63014)] Task finished successfully.
# 2023-10-11 20:13:45.636 [1697054507139036/end/5 (pid 63289)] Task is starting.
# 2023-10-11 20:13:47.707 [1697054507139036/end/5 (pid 63289)] done! great work!
# 2023-10-11 20:13:48.081 [1697054507139036/end/5 (pid 63289)] Task finished successfully.
# 2023-10-11 20:13:48.081 Done!