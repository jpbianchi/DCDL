"""
Flow #3: This flow will train a small (linear) neural network 
on the MNIST dataset to performance classification, and run an 
directionality test to measure model robustness.
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
from src.tests.directionality import MNISTDirectionalityTest
from src.utils import load_config, to_json


class DigitClassifierFlow(FlowSpec):
  r"""A flow that trains a image classifier to recognize handwritten
  digit, such as those in the MNIST dataset. Includes a directionality 
  test on transformed digits to test model robustness.

  Arguments
  ---------
  config (str, default: ./config.py): path to a configuration file
  """
  config_path = Parameter('config', 
    help = 'path to config file', default='./configs/directionality_flow.json')

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
      f'logs/directionality_flow', 'offline-test-results.json')

    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save to disk

    self.next(self.directionality_test)

  @step
  def directionality_test(self):
    r"""Runs an directionality test. Saves results to a log file."""

    test = MNISTDirectionalityTest()
    test.test(self.trainer, self.system)

    results = self.system.test_results
    pprint(results)

    log_file = join(Path(__file__).resolve().parent.parent, 
      f'logs/directionality_flow', 'directionality-test-results.json')

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
  To validate this flow, run `python directionality_flow.py`. To list
  this flow, run `python directionality_flow.py show`. To execute
  this flow, run `python directionality_flow.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python directionality_flow.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python directionality_flow.py resume`
  
  You can specify a run id as well.
  """
  flow = DigitClassifierFlow()


# gitpod /workspace/data-centric-deep-learning/course/week2/pipeline_project (main) $ python flows/directionality_flow.py run
# Metaflow 2.6.0 executing DigitClassifierFlow for user:gitpod
# Validating your flow...
#     The graph looks good!
# Running pylint...
#     Pylint is happy!
# 2023-09-26 23:01:31.039 Workflow starting (run-id 1695769291034540):
# 2023-09-26 23:01:31.063 [1695769291034540/start/1 (pid 277155)] Task is starting.
# 2023-09-26 23:01:33.544 [1695769291034540/start/1 (pid 277155)] Task finished successfully.
# 2023-09-26 23:01:33.567 [1695769291034540/init_system/2 (pid 277201)] Task is starting.
# 2023-09-26 23:01:35.933 [1695769291034540/init_system/2 (pid 277201)] GPU available: False, used: False
# 2023-09-26 23:01:35.934 [1695769291034540/init_system/2 (pid 277201)] TPU available: False, using: 0 TPU cores
# 2023-09-26 23:01:35.934 [1695769291034540/init_system/2 (pid 277201)] IPU available: False, using: 0 IPUs
# 2023-09-26 23:01:37.584 [1695769291034540/init_system/2 (pid 277201)] HPU available: False, using: 0 HPUs
# 2023-09-26 23:01:37.585 [1695769291034540/init_system/2 (pid 277201)] Task finished successfully.
# 2023-09-26 23:01:37.611 [1695769291034540/train_model/3 (pid 277257)] Task is starting.
# 2023-09-26 23:01:40.318 [1695769291034540/train_model/3 (pid 277257)] 
# Epoch 1:  80%|████████  | 1500/1875 [00:14<00:03, 104.77it/s, loss=0.276, v_num=7, train_loss=0.458, train_acc=0.906, dev_loss=0.275, dev_acc=0.917] ] 
# 2023-09-26 23:02:08.020 [1695769291034540/train_model/3 (pid 277257)] | Name  | Type       | Params|██████████| 375/375 [00:03<00:00, 97.50it/s] 
# 2023-09-26 23:02:08.020 [1695769291034540/train_model/3 (pid 277257)] -------------------------------------
# 2023-09-26 23:02:08.020 [1695769291034540/train_model/3 (pid 277257)] 0 | model | Sequential | 12.7 K
# 2023-09-26 23:02:08.020 [1695769291034540/train_model/3 (pid 277257)] -------------------------------------
# 2023-09-26 23:02:08.020 [1695769291034540/train_model/3 (pid 277257)] 12.7 K    Trainable params
# 2023-09-26 23:02:08.021 [1695769291034540/train_model/3 (pid 277257)] 0         Non-trainable params
# 2023-09-26 23:02:08.021 [1695769291034540/train_model/3 (pid 277257)] 12.7 K    Total params
# 2023-09-26 23:02:08.021 [1695769291034540/train_model/3 (pid 277257)] 0.051     Total estimated model params size (MB)
# 2023-09-26 23:02:08.021 [1695769291034540/train_model/3 (pid 277257)] Epoch 0, global step 1500: 'dev_loss' reached 0.27525 (best 0.27525), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/direEpoch 2:  80%|████████  | 1500/1875 [00:09<00:02, 151.98it/s, loss=0.215, v_num=7, train_loss=0.117, train_acc=0.938, dev_loss=0.249, dev_acc=0.926] 6]
# 2023-09-26 23:02:21.530 [1695769291034540/train_model/3 (pid 277257)] Epoch 1, global step 3000: 'dev_loss' reached 0.24901 (best 0.24901), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/direEpoch 3:  80%|████████  | 1500/1875 [00:09<00:02, 151.14it/s, loss=0.231, v_num=7, train_loss=0.0559, train_acc=1.000, dev_loss=0.234, dev_acc=0.934]]] 
# 2023-09-26 23:02:34.968 [1695769291034540/train_model/3 (pid 277257)] Epoch 2, global step 4500: 'dev_loss' reached 0.23427 (best 0.23427), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/direEpoch 4:  80%|████████  | 1500/1875 [00:09<00:02, 151.61it/s, loss=0.18, v_num=7, train_loss=0.380, train_acc=0.938, dev_loss=0.226, dev_acc=0.935]5]5]
# 2023-09-26 23:02:48.267 [1695769291034540/train_model/3 (pid 277257)] Epoch 3, global step 6000: 'dev_loss' reached 0.22644 (best 0.22644), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/direEpoch 5:  80%|████████  | 1500/1875 [00:09<00:02, 152.57it/s, loss=0.187, v_num=7, train_loss=0.168, train_acc=0.938, dev_loss=0.238, dev_acc=0.929] 9]
# Epoch 6:  80%|████████  | 1500/1875 [00:10<00:02, 145.08it/s, loss=0.228, v_num=7, train_loss=0.130, train_acc=0.938, dev_loss=0.243, dev_acc=0.930] ]
# Epoch 7:  80%|████████  | 1500/1875 [00:09<00:02, 151.48it/s, loss=0.184, v_num=7, train_loss=0.345, train_acc=0.906, dev_loss=0.235, dev_acc=0.929]   
# Epoch 8:  80%|████████  | 1500/1875 [00:09<00:02, 153.89it/s, loss=0.169, v_num=7, train_loss=0.073, train_acc=0.969, dev_loss=0.222, dev_acc=0.937] ]
# 2023-09-26 23:03:41.885 [1695769291034540/train_model/3 (pid 277257)] Epoch 7, global step 12000: 'dev_loss' reached 0.22176 (best 0.22176), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/dirEpoch 9:  80%|████████  | 1500/1875 [00:09<00:02, 152.80it/s, loss=0.189, v_num=7, train_loss=0.132, train_acc=0.938, dev_loss=0.216, dev_acc=0.937]  ]
# 2023-09-26 23:03:55.163 [1695769291034540/train_model/3 (pid 277257)] Epoch 8, global step 13500: 'dev_loss' reached 0.21620 (best 0.21620), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/dirEpoch 9: 100%|██████████| 1875/1875 [00:13<00:00, 142.06it/s, loss=0.189, v_num=7, train_loss=0.132, train_acc=0.938, dev_loss=0.208, dev_acc=0.940]
# 2023-09-26 23:03:58.525 [1695769291034540/train_model/3 (pid 277257)] Epoch 9, global step 15000: 'dev_loss' reached 0.20839 (best 0.20839), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/directionality_flow/epoch=9-step=15000.ckpt' as top 1
# 2023-09-26 23:04:02.985 [1695769291034540/train_model/3 (pid 277257)] Task finished successfully.
# 2023-09-26 23:04:03.020 [1695769291034540/offline_test/4 (pid 367915)] Task is starting.
# 2023-09-26 23:04:06.726 [1695769291034540/offline_test/4 (pid 367915)] Restoring states from the checkpoint path at /workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/directionality_flow/epoch=9-step=15000.ckpt
# 2023-09-26 23:04:06.736 [1695769291034540/offline_test/4 (pid 367915)] Loaded model weights from checkpoint at /workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/directionality_flow/epoch=9-step=15000.ckpt
# Testing DataLoader 0: 100%|██████████| 313/313 [00:02<00:00, 120.25it/s]s]ting: 0it [00:00, ?it/s]
# 2023-09-26 23:04:09.347 [1695769291034540/offline_test/4 (pid 367915)] {'acc': 0.9415934681892395, 'loss': 0.21347284317016602}
# 2023-09-26 23:04:13.779 [1695769291034540/offline_test/4 (pid 367915)] Task finished successfully.
# 2023-09-26 23:04:13.805 [1695769291034540/directionality_test/5 (pid 367985)] Task is starting.
# 100%|██████████| 10/10 [00:00<00:00, 2236.01it/s]onality_test/5 (pid 367985)] 0%|          | 0/10 [00:00<?, ?it/s]
# 2023-09-26 23:04:17.077 [1695769291034540/directionality_test/5 (pid 367985)] {'acc': 0.6899999976158142}               << ACC 0.67 NOT GOOD!
# 2023-09-26 23:04:20.170 [1695769291034540/directionality_test/5 (pid 367985)] Task finished successfully.
# 2023-09-26 23:04:20.197 [1695769291034540/end/6 (pid 368045)] Task is starting.
# 2023-09-26 23:04:22.439 [1695769291034540/end/6 (pid 368045)] done! great work!
# 2023-09-26 23:04:22.779 [1695769291034540/end/6 (pid 368045)] Task finished successfully.
# 2023-09-26 23:04:22.780 Done!