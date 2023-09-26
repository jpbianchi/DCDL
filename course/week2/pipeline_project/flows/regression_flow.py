"""
Flow #3: This flow will train a small (linear) neural network 
on the MNIST dataset to performance classification, and run an 
regression test to measure when a model is struggling.
"""

import os
import torch
import random
import numpy as np
from os.path import join
from pathlib import Path
from pprint import pprint

from metaflow import FlowSpec, step, Parameter

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.system import MNISTDataModule, DigitClassifierSystem
from src.tests.regression import MNISTRegressionTest
from src.utils import load_config, to_json


class DigitClassifierFlow(FlowSpec):
  r"""A flow that trains a image classifier to recognize handwritten
  digit, such as those in the MNIST dataset. Includes an regression 
  test using a trained model.

  Arguments
  ---------
  config (str, default: ./config.py): path to a configuration file
  """
  config_path = Parameter('config', 
    help = 'path to config file', default='./configs/regression_flow.json')

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
    config = load_config(self.config_path)

    dm = MNISTDataModule(config)
    system = DigitClassifierSystem(config)

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

    self.dm = dm
    self.system = system
    self.trainer = trainer

    self.next(self.train_model)

  @step
  def train_model(self):
    """Calls `fit` on the trainer."""

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
      f'logs/regression_flow', 'offline-test-results.json')

    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save to disk

    self.next(self.regression_test)

  @step
  def regression_test(self):
    r"""Runs an integration test. Saves results to a log file."""

    test = MNISTRegressionTest()
    test.test(self.trainer, self.system)

    results = self.system.test_results
    pprint(results)

    log_file = join(Path(__file__).resolve().parent.parent, 
      f'logs/regression_flow', 'regression-test-results.json')

    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)

    self.results = results
    self.next(self.end)

  @step
  def end(self):
    """End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python regression_flow.py`. To list
  this flow, run `python regression_flow.py show`. To execute
  this flow, run `python regression_flow.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python regression_flow.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python regression_flow.py resume`
  
  You can specify a run id as well.
  """
  flow = DigitClassifierFlow()

# first do python src/tests/regression.py artifacts/ckpts/train_flow/epoch=5-step=9000.ckpt

# gitpod /workspace/data-centric-deep-learning/course/week2/pipeline_project (main) $ python flows/regression_flow.py run
# Metaflow 2.6.0 executing DigitClassifierFlow for user:gitpod
# Validating your flow...
#     The graph looks good!
# Running pylint...
#     Pylint is happy!
# 2023-09-26 22:52:40.806 Workflow starting (run-id 1695768760801694):
# 2023-09-26 22:52:40.830 [1695768760801694/start/1 (pid 185400)] Task is starting.
# 2023-09-26 22:52:42.911 [1695768760801694/start/1 (pid 185400)] Task finished successfully.
# 2023-09-26 22:52:42.937 [1695768760801694/init_system/2 (pid 185426)] Task is starting.
# 2023-09-26 22:52:44.874 [1695768760801694/init_system/2 (pid 185426)] GPU available: False, used: False
# 2023-09-26 22:52:46.535 [1695768760801694/init_system/2 (pid 185426)] TPU available: False, using: 0 TPU cores
# 2023-09-26 22:52:46.535 [1695768760801694/init_system/2 (pid 185426)] IPU available: False, using: 0 IPUs
# 2023-09-26 22:52:46.535 [1695768760801694/init_system/2 (pid 185426)] HPU available: False, using: 0 HPUs
# 2023-09-26 22:52:46.537 [1695768760801694/init_system/2 (pid 185426)] Task finished successfully.
# 2023-09-26 22:52:46.562 [1695768760801694/train_model/3 (pid 185464)] Task is starting.
# 2023-09-26 22:52:48.972 [1695768760801694/train_model/3 (pid 185464)] 
# Epoch 1:  80%|████████  | 1500/1875 [00:09<00:02, 156.15it/s, loss=0.193, v_num=6, train_loss=0.197, train_acc=0.906, dev_loss=0.275, dev_acc=0.921] 1]
# 2023-09-26 22:53:11.571 [1695768760801694/train_model/3 (pid 185464)] | Name  | Type       | Params|██████████| 375/375 [00:03<00:00, 113.18it/s]
# 2023-09-26 22:53:11.571 [1695768760801694/train_model/3 (pid 185464)] -------------------------------------
# 2023-09-26 22:53:11.571 [1695768760801694/train_model/3 (pid 185464)] 0 | model | Sequential | 12.7 K
# 2023-09-26 22:53:11.571 [1695768760801694/train_model/3 (pid 185464)] -------------------------------------
# 2023-09-26 22:53:11.571 [1695768760801694/train_model/3 (pid 185464)] 12.7 K    Trainable params
# 2023-09-26 22:53:11.572 [1695768760801694/train_model/3 (pid 185464)] 0         Non-trainable params
# 2023-09-26 22:53:11.572 [1695768760801694/train_model/3 (pid 185464)] 12.7 K    Total params
# 2023-09-26 22:53:11.572 [1695768760801694/train_model/3 (pid 185464)] 0.051     Total estimated model params size (MB)
# 2023-09-26 22:53:11.572 [1695768760801694/train_model/3 (pid 185464)] Epoch 0, global step 1500: 'dev_loss' reached 0.27479 (best 0.27479), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/regrEpoch 2:  80%|████████  | 1500/1875 [00:09<00:02, 155.55it/s, loss=0.166, v_num=6, train_loss=0.0643, train_acc=1.000, dev_loss=0.254, dev_acc=0.927]
# 2023-09-26 22:53:24.600 [1695768760801694/train_model/3 (pid 185464)] Epoch 1, global step 3000: 'dev_loss' reached 0.25410 (best 0.25410), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/regrEpoch 3:  80%|████████  | 1500/1875 [00:09<00:02, 156.19it/s, loss=0.181, v_num=6, train_loss=0.0709, train_acc=1.000, dev_loss=0.258, dev_acc=0.928]]
# Epoch 4:  80%|████████  | 1500/1875 [00:10<00:02, 149.85it/s, loss=0.169, v_num=6, train_loss=0.278, train_acc=0.906, dev_loss=0.233, dev_acc=0.934] ]
# 2023-09-26 22:53:50.875 [1695768760801694/train_model/3 (pid 185464)] Epoch 3, global step 6000: 'dev_loss' reached 0.23304 (best 0.23304), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/regrEpoch 5:  80%|████████  | 1500/1875 [00:09<00:02, 155.98it/s, loss=0.173, v_num=6, train_loss=0.0779, train_acc=0.969, dev_loss=0.236, dev_acc=0.933]3]]
# Epoch 6:  80%|████████  | 1500/1875 [00:09<00:02, 155.87it/s, loss=0.18, v_num=6, train_loss=0.111, train_acc=0.938, dev_loss=0.240, dev_acc=0.932]2] ]
# Epoch 7:  80%|████████  | 1500/1875 [00:09<00:02, 151.86it/s, loss=0.174, v_num=6, train_loss=0.125, train_acc=0.969, dev_loss=0.219, dev_acc=0.938]8]]
# 2023-09-26 22:54:29.942 [1695768760801694/train_model/3 (pid 185464)] Epoch 6, global step 10500: 'dev_loss' reached 0.21908 (best 0.21908), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/regEpoch 8:  80%|████████  | 1500/1875 [00:09<00:02, 155.70it/s, loss=0.177, v_num=6, train_loss=0.160, train_acc=0.938, dev_loss=0.220, dev_acc=0.937]  
# Epoch 9:  80%|████████  | 1500/1875 [00:09<00:02, 153.99it/s, loss=0.171, v_num=6, train_loss=0.149, train_acc=0.938, dev_loss=0.226, dev_acc=0.939]  
# Epoch 9: 100%|██████████| 1875/1875 [00:13<00:00, 142.84it/s, loss=0.171, v_num=6, train_loss=0.149, train_acc=0.938, dev_loss=0.219, dev_acc=0.940]
# 2023-09-26 22:54:59.442 [1695768760801694/train_model/3 (pid 185464)] Epoch 9, global step 15000: 'dev_loss' reached 0.21872 (best 0.21872), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/regression_flow/epoch=9-step=15000.ckpt' as top 1
# 2023-09-26 22:55:04.103 [1695768760801694/train_model/3 (pid 185464)] Task finished successfully.
# 2023-09-26 22:55:04.151 [1695768760801694/offline_test/4 (pid 275845)] Task is starting.
# 2023-09-26 22:55:07.879 [1695768760801694/offline_test/4 (pid 275845)] Restoring states from the checkpoint path at /workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/regression_flow/epoch=9-step=15000.ckpt
# 2023-09-26 22:55:07.894 [1695768760801694/offline_test/4 (pid 275845)] Loaded model weights from checkpoint at /workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/regression_flow/epoch=9-step=15000.ckpt
# Testing DataLoader 0: 100%|██████████| 313/313 [00:03<00:00, 99.60it/s] esting: 0it [00:00, ?it/s]
# 2023-09-26 22:55:11.045 [1695768760801694/offline_test/4 (pid 275845)] {'acc': 0.9427915215492249, 'loss': 0.21832221746444702}
# 2023-09-26 22:55:15.739 [1695768760801694/offline_test/4 (pid 275845)] Task finished successfully.
# 2023-09-26 22:55:15.767 [1695768760801694/regression_test/5 (pid 275942)] Task is starting.
# Testing DataLoader 0: 100%|██████████| 10/10 [00:00<00:00, 830.05it/s]2)] Testing: 0it [00:00, ?it/s]
# 2023-09-26 22:55:18.963 [1695768760801694/regression_test/5 (pid 275942)] {'acc': 0.949999988079071, 'loss': 0.13009703159332275}  <<< ACC 0.95
# 2023-09-26 22:55:22.238 [1695768760801694/regression_test/5 (pid 275942)] Task finished successfully.
# 2023-09-26 22:55:22.271 [1695768760801694/end/6 (pid 275991)] Task is starting.
# 2023-09-26 22:55:24.324 [1695768760801694/end/6 (pid 275991)] done! great work!
# 2023-09-26 22:55:24.656 [1695768760801694/end/6 (pid 275991)] Task finished successfully.
# 2023-09-26 22:55:24.657 Done!