"""
Flow #3: This flow will train a small (linear) neural network 
on the MNIST dataset to performance classification, and run an 
integration test to measure model performance in-the-wild.
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
from src.tests.integration import MNISTIntegrationTest
from src.utils import load_config, to_json


class DigitClassifierFlow(FlowSpec):
  r"""A flow that trains a image classifier to recognize handwritten
  digit, such as those in the MNIST dataset. Includes an integration 
  test on handwritten digits provided by you!

  Arguments
  ---------
  config (str, default: ./config.py): path to a configuration file
  """
  config_path = Parameter('config', 
    help = 'path to config file', default='./configs/integration_flow.json')

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
      f'logs/integration_flow', 'offline-test-results.json')

    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save to disk

    self.next(self.integration_test)

  @step
  def integration_test(self):
    r"""Runs an integration test. Saves results to a log file."""

    test = MNISTIntegrationTest()
    test.test(self.trainer, self.system)

    results = self.system.test_results
    pprint(results)

    log_file = join(Path(__file__).resolve().parent.parent, 
      f'logs/integration_flow', 'integration-test-results.json')

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
  To validate this flow, run `python integration_flow.py`. To list
  this flow, run `python integration_flow.py show`. To execute
  this flow, run `python integration_flow.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python integration_flow.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python integration_flow.py resume`
  
  You can specify a run id as well.
  """
  flow = DigitClassifierFlow()

# gitpod /workspace/data-centric-deep-learning/course/week2/pipeline_project (main) $ python flows/integration_flow.py run
# Metaflow 2.6.0 executing DigitClassifierFlow for user:gitpod
# Validating your flow...
#     The graph looks good!
# Running pylint...
#     Pylint is happy!
# 2023-09-26 22:26:57.946 Workflow starting (run-id 1695767217941323):
# 2023-09-26 22:26:57.972 [1695767217941323/start/1 (pid 180690)] Task is starting.
# 2023-09-26 22:27:00.354 [1695767217941323/start/1 (pid 180690)] Task finished successfully.
# 2023-09-26 22:27:00.379 [1695767217941323/init_system/2 (pid 180715)] Task is starting.
# 2023-09-26 22:27:02.655 [1695767217941323/init_system/2 (pid 180715)] GPU available: False, used: False
# 2023-09-26 22:27:02.656 [1695767217941323/init_system/2 (pid 180715)] TPU available: False, using: 0 TPU cores
# 2023-09-26 22:27:02.656 [1695767217941323/init_system/2 (pid 180715)] IPU available: False, using: 0 IPUs
# 2023-09-26 22:27:02.656 [1695767217941323/init_system/2 (pid 180715)] HPU available: False, using: 0 HPUs
# 2023-09-26 22:27:04.340 [1695767217941323/init_system/2 (pid 180715)] Task finished successfully.
# 2023-09-26 22:27:04.367 [1695767217941323/train_model/3 (pid 180750)] Task is starting.
# 2023-09-26 22:27:07.042 [1695767217941323/train_model/3 (pid 180750)] 
# Epoch 1:  80%|████████  | 1500/1875 [00:19<00:04, 78.91it/s, loss=0.274, v_num=4, train_loss=0.159, train_acc=0.938, dev_loss=0.310, dev_acc=0.909] ]]
# 2023-09-26 22:27:48.544 [1695767217941323/train_model/3 (pid 180750)] | Name  | Type   | Params100%|██████████| 375/375 [00:03<00:00, 124.76it/s]
# 2023-09-26 22:27:48.544 [1695767217941323/train_model/3 (pid 180750)] ---------------------------------
# 2023-09-26 22:27:48.545 [1695767217941323/train_model/3 (pid 180750)] 0 | model | Linear | 7.9 K | 0/375 [00:00<?, ?it/s]
# Epoch 1:  80%|████████  | 1501/1875 [00:19<00:04, 78.93it/s, loss=0.274, v_num=4, train_loss=0.159, train_acc=0.938, dev_loss=0.310, dev_acc=0.909]
# Epoch 1:  80%|████████  | 1502/1875 [00:19<00:04, 78.97it/s, loss=0.274, v_num=4, train_loss=0.159, train_acc=0.938, dev_loss=0.310, dev_acc=0.909]
# 2023-09-26 22:27:48.545 [1695767217941323/train_model/3 (pid 180750)] 0         Non-trainable params          | 2/375 [00:00<00:00, 390.04it/s] 
# 2023-09-26 22:27:48.545 [1695767217941323/train_model/3 (pid 180750)] 7.9 K     Total params
# 2023-09-26 22:27:48.546 [1695767217941323/train_model/3 (pid 180750)] 0.031     Total estimated model params size (MB)
# 2023-09-26 22:27:48.546 [1695767217941323/train_model/3 (pid 180750)] Epoch 0, global step 1500: 'dev_loss' reached 0.31030 (best 0.31030), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/inteEpoch 2:  80%|████████  | 1500/1875 [00:19<00:04, 78.53it/s, loss=0.386, v_num=4, train_loss=0.290, train_acc=0.906, dev_loss=0.285, dev_acc=0.918]8]
# 2023-09-26 22:28:11.115 [1695767217941323/train_model/3 (pid 180750)] Epoch 1, global step 3000: 'dev_loss' reached 0.28506 (best 0.28506), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/inteEpoch 3:  80%|████████  | 1500/1875 [00:18<00:04, 79.26it/s, loss=0.292, v_num=4, train_loss=0.298, train_acc=0.906, dev_loss=0.299, dev_acc=0.914]]4]
# Epoch 4:  80%|████████  | 1500/1875 [00:19<00:04, 78.01it/s, loss=0.29, v_num=4, train_loss=0.450, train_acc=0.938, dev_loss=0.285, dev_acc=0.921]] ]
# Epoch 5:  80%|████████  | 1500/1875 [00:19<00:04, 78.06it/s, loss=0.229, v_num=4, train_loss=0.270, train_acc=0.906, dev_loss=0.296, dev_acc=0.917] ]
# Epoch 6:  80%|████████  | 1500/1875 [00:19<00:04, 76.88it/s, loss=0.275, v_num=4, train_loss=0.368, train_acc=0.906, dev_loss=0.304, dev_acc=0.918] 
# Epoch 7:  80%|████████  | 1500/1875 [00:18<00:04, 79.79it/s, loss=0.264, v_num=4, train_loss=0.220, train_acc=0.844, dev_loss=0.292, dev_acc=0.919] ]19]
# Epoch 8:  80%|████████  | 1500/1875 [00:19<00:04, 78.80it/s, loss=0.248, v_num=4, train_loss=0.131, train_acc=0.938, dev_loss=0.300, dev_acc=0.916] ]
# Epoch 9:  80%|████████  | 1500/1875 [00:19<00:04, 78.18it/s, loss=0.246, v_num=4, train_loss=0.367, train_acc=0.875, dev_loss=0.297, dev_acc=0.921] 1]
# Epoch 9: 100%|██████████| 1875/1875 [00:22<00:00, 81.98it/s, loss=0.246, v_num=4, train_loss=0.367, train_acc=0.875, dev_loss=0.303, dev_acc=0.915]
# 2023-09-26 22:30:52.716 [1695767217941323/train_model/3 (pid 180750)] Epoch 9, global step 15000: 'dev_loss' was not in top 13<00:00, 101.04it/s]
# 2023-09-26 22:30:57.138 [1695767217941323/train_model/3 (pid 180750)] Task finished successfully.
# 2023-09-26 22:30:57.176 [1695767217941323/offline_test/4 (pid 181347)] Task is starting.
# 2023-09-26 22:31:00.853 [1695767217941323/offline_test/4 (pid 181347)] Restoring states from the checkpoint path at /workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/integration_flow/epoch=1-step=3000.ckpt
# 2023-09-26 22:31:00.868 [1695767217941323/offline_test/4 (pid 181347)] Loaded model weights from checkpoint at /workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/integration_flow/epoch=1-step=3000.ckpt
# Testing DataLoader 0: 100%|██████████| 313/313 [00:03<00:00, 83.10it/s]Testing: 0it [00:00, ?it/s]
# 2023-09-26 22:31:04.643 [1695767217941323/offline_test/4 (pid 181347)] {'acc': 0.9216253757476807, 'loss': 0.28261256217956543}
# 2023-09-26 22:31:09.216 [1695767217941323/offline_test/4 (pid 181347)] Task finished successfully.
# 2023-09-26 22:31:09.250 [1695767217941323/integration_test/5 (pid 181411)] Task is starting.
# 2023-09-26 22:31:12.414 [1695767217941323/integration_test/5 (pid 181411)] {'acc': 0.9216253757476807, 'loss': 0.28261256217956543}  <<< SAME ACCURACY AS OFFLINE TEST
# 2023-09-26 22:31:15.498 [1695767217941323/integration_test/5 (pid 181411)] Task finished successfully.
# 2023-09-26 22:31:15.525 [1695767217941323/end/6 (pid 181457)] Task is starting.
# 2023-09-26 22:31:17.520 [1695767217941323/end/6 (pid 181457)] done! great work!
# 2023-09-26 22:31:17.827 [1695767217941323/end/6 (pid 181457)] Task finished successfully.
# 2023-09-26 22:31:17.828 Done!