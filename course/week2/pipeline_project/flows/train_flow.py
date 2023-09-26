"""
Flow #1: This flow will train a small (linear) neural network 
on the MNIST dataset to performance classification.
"""

import os
import torch
import random
import shutil
import numpy as np
from os.path import join
from pathlib import Path
from pprint import pprint

from metaflow import FlowSpec, step, Parameter

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.system import MNISTDataModule, DigitClassifierSystem
from src.utils import load_config, to_json


class DigitClassifierFlow(FlowSpec):
  r"""A flow that trains a image classifier to recognize handwritten
  digit, such as those in the MNIST dataset.

  Arguments
  ---------
  config (str, default: ./config.py): path to a configuration file
  """
  config_path = Parameter('config', 
    help = 'path to config file', default='./configs/train_flow.json')

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
    dm = MNISTDataModule(config)

    # a PyTorch Lightning system wraps around model logic
    system = DigitClassifierSystem(config)

    # a callback to save best model weights
    checkpoint_callback = ModelCheckpoint(
      dirpath = config.system.save_dir,
      monitor = 'dev_loss',
      mode = 'min',    # look for lowest `dev_loss`
      save_top_k = 1,  # save top 1 checkpoints
      verbose = True,
    )

    trainer = Trainer(
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

    log_file = join(Path(__file__).resolve().parent.parent, 
      f'logs/train_flow', 'offline-test-results.json')

    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save to disk

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
  flow = DigitClassifierFlow()
  
  
# gitpod /workspace/data-centric-deep-learning/course/week2/pipeline_project (main) $ python flows/train_flow.py run
# Metaflow 2.6.0 executing DigitClassifierFlow for user:gitpod
# Validating your flow...
#     The graph looks good!
# Running pylint...
#     Pylint is happy!
# 2023-09-26 21:29:26.112 Workflow starting (run-id 1695763766106396):
# 2023-09-26 21:29:26.135 [1695763766106396/start/1 (pid 19210)] Task is starting.
# 2023-09-26 21:29:28.221 [1695763766106396/start/1 (pid 19210)] Task finished successfully.
# 2023-09-26 21:29:28.245 [1695763766106396/init_system/2 (pid 19235)] Task is starting.
# 2023-09-26 21:29:30.220 [1695763766106396/init_system/2 (pid 19235)] GPU available: False, used: False
# 2023-09-26 21:29:30.220 [1695763766106396/init_system/2 (pid 19235)] TPU available: False, using: 0 TPU cores
# 2023-09-26 21:29:30.221 [1695763766106396/init_system/2 (pid 19235)] IPU available: False, using: 0 IPUs
# 2023-09-26 21:29:30.221 [1695763766106396/init_system/2 (pid 19235)] HPU available: False, using: 0 HPUs
# 2023-09-26 21:29:31.840 [1695763766106396/init_system/2 (pid 19235)] Task finished successfully.
# 2023-09-26 21:29:31.862 [1695763766106396/train_model/3 (pid 19279)] Task is starting.
# 2023-09-26 21:29:34.240 [1695763766106396/train_model/3 (pid 19279)] Missing logger folder: /workspace/data-centric-deep-learning/course/week2/pipeline_project/lightning_logs
# 2023-09-26 21:29:34.242 [1695763766106396/train_model/3 (pid 19279)] 
# Epoch 1:  80%|████████  | 1500/1875 [00:26<00:06, 57.05it/s, loss=0.318, v_num=0, train_loss=0.213, train_acc=0.938, dev_loss=0.312, dev_acc=0.914] ]
# 2023-09-26 21:30:22.346 [1695763766106396/train_model/3 (pid 19279)] | Name  | Type   | Params100%|██████████| 375/375 [00:03<00:00, 88.20it/s] 
# 2023-09-26 21:30:22.346 [1695763766106396/train_model/3 (pid 19279)] ---------------------------------
# 2023-09-26 21:30:22.347 [1695763766106396/train_model/3 (pid 19279)] 0 | model | Linear | 7.9 K
# 2023-09-26 21:30:22.347 [1695763766106396/train_model/3 (pid 19279)] ---------------------------------
# 2023-09-26 21:30:22.347 [1695763766106396/train_model/3 (pid 19279)] 7.9 K     Trainable params
# 2023-09-26 21:30:22.347 [1695763766106396/train_model/3 (pid 19279)] 0         Non-trainable params
# 2023-09-26 21:30:22.347 [1695763766106396/train_model/3 (pid 19279)] 7.9 K     Total params
# 2023-09-26 21:30:22.347 [1695763766106396/train_model/3 (pid 19279)] 0.031     Total estimated model params size (MB)
# 2023-09-26 21:30:22.348 [1695763766106396/train_model/3 (pid 19279)] Epoch 0, global step 1500: 'dev_loss' reached 0.31238 (best 0.31238), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/train_flow/epoch=0-sEpoch 2:  80%|████████  | 1500/1875 [00:18<00:04, 81.93it/s, loss=0.292, v_num=0, train_loss=0.118, train_acc=0.969, dev_loss=0.295, dev_acc=0.919] ]
# 2023-09-26 21:30:44.115 [1695763766106396/train_model/3 (pid 19279)] Epoch 1, global step 3000: 'dev_loss' reached 0.29543 (best 0.29543), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/train_flow/epoch=1-sEpoch 3:  80%|████████  | 1500/1875 [00:18<00:04, 82.08it/s, loss=0.236, v_num=0, train_loss=0.0982, train_acc=0.969, dev_loss=0.295, dev_acc=0.919]
# 2023-09-26 21:31:05.744 [1695763766106396/train_model/3 (pid 19279)] Epoch 2, global step 4500: 'dev_loss' reached 0.29478 (best 0.29478), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/train_flow/epoch=2-sEpoch 4:  80%|████████  | 1500/1875 [00:18<00:04, 79.46it/s, loss=0.231, v_num=0, train_loss=0.329, train_acc=0.844, dev_loss=0.292, dev_acc=0.924] ]
# 2023-09-26 21:31:27.947 [1695763766106396/train_model/3 (pid 19279)] Epoch 3, global step 6000: 'dev_loss' reached 0.29169 (best 0.29169), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/train_flow/epoch=3-sEpoch 5:  80%|████████  | 1500/1875 [00:18<00:04, 80.15it/s, loss=0.278, v_num=0, train_loss=0.170, train_acc=0.969, dev_loss=0.309, dev_acc=0.916] 6]
# Epoch 6:  80%|████████  | 1500/1875 [00:18<00:04, 79.43it/s, loss=0.287, v_num=0, train_loss=0.723, train_acc=0.812, dev_loss=0.292, dev_acc=0.922] ]
# 2023-09-26 21:32:12.232 [1695763766106396/train_model/3 (pid 19279)] Epoch 5, global step 9000: 'dev_loss' reached 0.29155 (best 0.29155), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/train_flow/epoch=5-sEpoch 7:  80%|████████  | 1500/1875 [00:18<00:04, 79.45it/s, loss=0.27, v_num=0, train_loss=0.167, train_acc=0.938, dev_loss=0.303, dev_acc=0.916]] ]
# Epoch 8:  80%|████████  | 1500/1875 [00:18<00:04, 81.59it/s, loss=0.301, v_num=0, train_loss=0.0896, train_acc=0.969, dev_loss=0.316, dev_acc=0.914]]
# Epoch 9:  80%|████████  | 1500/1875 [00:18<00:04, 81.43it/s, loss=0.288, v_num=0, train_loss=0.515, train_acc=0.875, dev_loss=0.301, dev_acc=0.920] ]
# Epoch 9: 100%|██████████| 1875/1875 [00:21<00:00, 86.73it/s, loss=0.288, v_num=0, train_loss=0.515, train_acc=0.875, dev_loss=0.303, dev_acc=0.920]
# 2023-09-26 21:33:21.440 [1695763766106396/train_model/3 (pid 19279)] Epoch 9, global step 15000: 'dev_loss' was not in top 13<00:00, 120.55it/s]
# 2023-09-26 21:33:25.803 [1695763766106396/train_model/3 (pid 19279)] Task finished successfully.
# 2023-09-26 21:33:25.834 [1695763766106396/offline_test/4 (pid 20441)] Task is starting.
# 2023-09-26 21:33:29.143 [1695763766106396/offline_test/4 (pid 20441)] Restoring states from the checkpoint path at /workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/train_flow/epoch=5-step=9000.ckpt
# 2023-09-26 21:33:29.152 [1695763766106396/offline_test/4 (pid 20441)] Loaded model weights from checkpoint at /workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/train_flow/epoch=5-step=9000.ckpt
# Testing DataLoader 0: 100%|██████████| 313/313 [00:02<00:00, 123.11it/s]]ting: 0it [00:00, ?it/s]
# 2023-09-26 21:33:31.710 [1695763766106396/offline_test/4 (pid 20441)] {'acc': 0.9252196550369263, 'loss': 0.2859332859516144}  <<< FINAL RESULT acc 0.92
# 2023-09-26 21:33:36.029 [1695763766106396/offline_test/4 (pid 20441)] Task finished successfully.
# 2023-09-26 21:33:36.051 [1695763766106396/end/5 (pid 20498)] Task is starting.
# 2023-09-26 21:33:37.846 [1695763766106396/end/5 (pid 20498)] done! great work!
# 2023-09-26 21:33:38.107 [1695763766106396/end/5 (pid 20498)] Task finished successfully.