"""Flow implementing "Distributionally Robust Neural Network For Group 
Shifts: On the Importance of Regularization for Worst-Case Generalization". 
See https://arxiv.org/pdf/1911.08731.pdf.
"""
import os
import torch
import random 
import numpy as np
from pprint import pprint
from os.path import join
from metaflow import FlowSpec, step, Parameter

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.systems import ReviewDataModule, RobustSentimentSystem
from src.paths import LOG_DIR, CONFIG_DIR
from src.utils import load_config, to_json


class DistRobustOpt(FlowSpec):
  r"""A flow that implements Equation 4 on page 3 of the paper. 

  We assume access to group labels, meaning whether an example is in 
  English or Spanish (this should be quite easy to obtain). We do not 
  assume access to this in test sets. Then, we minimize the maximum 
  group loss over all group.
  """
  config_path = Parameter('config', 
    help = 'path to config file', default = join(CONFIG_DIR, 'dro.json'))
  
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
    dm = ReviewDataModule(config)

    # your implementation will be used here!
    system = RobustSentimentSystem(config)

    checkpoint_callback = ModelCheckpoint(
      dirpath = config.system.save_dir,
      save_last = True,  # save the last epoch!
      verbose = True,
    )

    trainer = Trainer(
      logger = TensorBoardLogger(save_dir=LOG_DIR),
      max_epochs = config.system.optimizer.max_epochs,
      callbacks = [checkpoint_callback])

    self.dm = dm
    self.system = system
    self.trainer = trainer

    self.next(self.train_dro)

  @step
  def train_dro(self):
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

    pprint(results)  # print results to command line

    log_file = join(LOG_DIR, 'dro_flow', 'results.json')
    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save to disk

    self.next(self.end)

  @step
  def end(self):
    """End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python dro_flow.py`. To list
  this flow, run `python dro_flow.py show`. To execute
  this flow, run `python dro_flow.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python dro_flow.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python dro_flow.py resume`
  
  You can specify a run id as well.
  """
  flow = DistRobustOpt()

# python dro_flow.py run --config configs/dro.json

# Metaflow 2.6.0 executing DistRobustOpt for user:gitpod
# Validating your flow...
#     The graph looks good!
# Running pylint...
#     Pylint is happy!
# 2023-10-11 20:36:59.068 Workflow starting (run-id 1697056619063715):
# 2023-10-11 20:36:59.092 [1697056619063715/start/1 (pid 67480)] Task is starting.
# 2023-10-11 20:37:01.897 [1697056619063715/start/1 (pid 67480)] Task finished successfully.
# 2023-10-11 20:37:01.923 [1697056619063715/init_system/2 (pid 67505)] Task is starting.
# 2023-10-11 20:37:06.096 [1697056619063715/init_system/2 (pid 67505)] GPU available: False, used: False
# 2023-10-11 20:37:06.097 [1697056619063715/init_system/2 (pid 67505)] TPU available: False, using: 0 TPU cores
# 2023-10-11 20:37:06.097 [1697056619063715/init_system/2 (pid 67505)] IPU available: False, using: 0 IPUs
# 2023-10-11 20:37:51.291 [1697056619063715/init_system/2 (pid 67505)] HPU available: False, using: 0 HPUs
# 2023-10-11 20:37:51.292 [1697056619063715/init_system/2 (pid 67505)] Task finished successfully.
# 2023-10-11 20:37:51.320 [1697056619063715/train_dro/3 (pid 67573)] Task is starting.
# 2023-10-11 20:38:03.337 [1697056619063715/train_dro/3 (pid 67573)] 
# Epoch 9: 100%|██████████| 1282/1282 [00:32<00:00, 39.78it/s, loss=0.965, v_num=63, train_loss=0.743, train_acc=0.875, dev_loss=0.970, dev_acc=0.834]4]
# 2023-10-11 20:46:17.402 [1697056619063715/train_dro/3 (pid 67573)] | Name  | Type       | Params|██████████| 32/32 [00:00<00:00, 51.88it/s] 
# 2023-10-11 20:46:17.403 [1697056619063715/train_dro/3 (pid 67573)] -------------------------------------
# 2023-10-11 20:46:17.403 [1697056619063715/train_dro/3 (pid 67573)] 0 | model | Sequential | 49.3 K
# 2023-10-11 20:46:17.403 [1697056619063715/train_dro/3 (pid 67573)] -------------------------------------
# 2023-10-11 20:46:17.403 [1697056619063715/train_dro/3 (pid 67573)] 49.3 K    Trainable params
# 2023-10-11 20:46:17.403 [1697056619063715/train_dro/3 (pid 67573)] 0         Non-trainable params
# 2023-10-11 20:46:17.403 [1697056619063715/train_dro/3 (pid 67573)] 49.3 K    Total params
# 2023-10-11 20:46:17.403 [1697056619063715/train_dro/3 (pid 67573)] 0.197     Total estimated model params size (MB)
# 2023-10-11 20:46:17.404 [1697056619063715/train_dro/3 (pid 67573)] Task finished successfully.
# 2023-10-11 20:46:17.434 [1697056619063715/offline_test/4 (pid 70201)] Task is starting.
# 2023-10-11 20:46:50.000 [1697056619063715/offline_test/4 (pid 70201)] Restoring states from the checkpoint path at /workspace/DCDL/course/week4/monitor_project/artifacts/ckpts/dro/epoch=9-step=12500.ckpt
# 2023-10-11 20:46:50.013 [1697056619063715/offline_test/4 (pid 70201)] Loaded model weights from checkpoint at /workspace/DCDL/course/week4/monitor_project/artifacts/ckpts/dro/epoch=9-step=12500.ckpt
# Testing DataLoader 0: 100%|██████████| 32/32 [00:00<00:00, 32.17it/s] Testing: 0it [00:00, ?it/s]
# 2023-10-11 20:46:51.723 [1697056619063715/offline_test/4 (pid 70201)] {'acc': 0.834, 'loss': 0.965}. # <<< ACC 0.834
# 2023-10-11 20:49:08.679 [1697056619063715/offline_test/4 (pid 70201)] Task finished successfully.
# 2023-10-11 20:49:08.706 [1697056619063715/end/5 (pid 70435)] Task is starting.
# 2023-10-11 20:49:10.815 [1697056619063715/end/5 (pid 70435)] done! great work!
# 2023-10-11 20:49:11.233 [1697056619063715/end/5 (pid 70435)] Task finished successfully.
# 2023-10-11 20:49:11.233 Done!

