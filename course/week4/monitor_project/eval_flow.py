"""Suppose we have collected test sets for both groups: english and 
spanish for the purpose of evaluation. 
"""

import os
import torch
import random
import numpy as np
from pprint import pprint
from os.path import join
from torch.utils.data import DataLoader
from metaflow import FlowSpec, step, Parameter

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from src.systems import SentimentClassifierSystem
from src.dataset import ProductReviewEmbeddings
from src.paths import LOG_DIR, CONFIG_DIR
from src.utils import load_config, to_json


class EvalClassifier(FlowSpec):
  r"""A flow that evaluates a trained sentiment classifier on sets
  of English and Spanish reviews. In the data distribution, these two
  groups are not evenly balanced. This flow serves as an evaluation 
  for how well the model does on each group individually.

  Arguments
  ---------
  config (str, default: ./configs/eval.json): path to a configuration file
  """
  config_path = Parameter('config', 
    help = 'path to config file', default = join(CONFIG_DIR, 'eval.json'))
  
  @step
  def start(self):
    r"""Start node.
    Set random seeds for reproducibility.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    self.next(self.load_system)

  @step
  def load_system(self):
    r"""Load pretrain system on new training data."""
    config = load_config(self.config_path)
    system = SentimentClassifierSystem.load_from_checkpoint(config.system.ckpt_path)
    trainer = Trainer(logger = TensorBoardLogger(save_dir=LOG_DIR))

    self.system = system
    self.trainer = trainer
  
    self.next(self.evaluate)

  @step
  def evaluate(self):
    r"""Evaluate system on two different test datasets."""

    config = load_config(self.config_path)
    en_ds = ProductReviewEmbeddings(lang='en', split='test')
    es_ds = ProductReviewEmbeddings(lang='es', split='test')

    en_dl = DataLoader(en_ds, batch_size = config.system.batch_size, 
      num_workers = config.system.num_workers)
    es_dl = DataLoader(es_ds, batch_size = config.system.batch_size, 
      num_workers = config.system.num_workers)

    self.trainer.test(self.system, dataloaders = en_dl)
    en_results = self.system.test_results

    self.trainer.test(self.system, dataloaders = es_dl)
    es_results = self.system.test_results

    print('Results on English reviews:')
    pprint(en_results)

    print('Results on Spanish reviews:')
    pprint(es_results)

    log_file = join(LOG_DIR, 'eval_flow', 'en_results.json')
    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(en_results, log_file)  # save to disk

    log_file = join(LOG_DIR, 'eval_flow', 'es_results.json')
    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(es_results, log_file)  # save to disk

    self.next(self.end)

  @step
  def end(self):
    """End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python eval_flow.py`. To list
  this flow, run `python eval_flow.py show`. To execute
  this flow, run `python eval_flow.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python eval_flow.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python eval_flow.py resume`
  
  You can specify a run id as well.
  """
  flow = EvalClassifier()

# python eval_flow.py run --config configs/eval.json

# Metaflow 2.6.0 executing EvalClassifier for user:gitpod
# Validating your flow...
#     The graph looks good!
# Running pylint...
#     Pylint is happy!
# 2023-10-11 20:20:48.085 Workflow starting (run-id 1697055648079786):
# 2023-10-11 20:20:48.105 [1697055648079786/start/1 (pid 64187)] Task is starting.
# 2023-10-11 20:20:50.420 [1697055648079786/start/1 (pid 64187)] Task finished successfully.
# 2023-10-11 20:20:50.440 [1697055648079786/load_system/2 (pid 64215)] Task is starting.
# 2023-10-11 20:20:52.460 [1697055648079786/load_system/2 (pid 64215)] GPU available: False, used: False
# 2023-10-11 20:20:52.460 [1697055648079786/load_system/2 (pid 64215)] TPU available: False, using: 0 TPU cores
# 2023-10-11 20:20:52.460 [1697055648079786/load_system/2 (pid 64215)] IPU available: False, using: 0 IPUs
# 2023-10-11 20:20:52.461 [1697055648079786/load_system/2 (pid 64215)] HPU available: False, using: 0 HPUs
# 2023-10-11 20:20:52.861 [1697055648079786/load_system/2 (pid 64215)] Task finished successfully.
# 2023-10-11 20:20:52.881 [1697055648079786/evaluate/3 (pid 64244)] Task is starting.
# Testing DataLoader 0: 100%|██████████| 32/32 [00:00<00:00, 54.96it/s]ting: 0it [00:00, ?it/s]
# Testing DataLoader 0: 100%|██████████| 32/32 [00:00<00:00, 53.61it/s]ting: 0it [00:00, ?it/s]
# 2023-10-11 20:20:57.219 [1697055648079786/evaluate/3 (pid 64244)] Results on English reviews:
# 2023-10-11 20:20:57.219 [1697055648079786/evaluate/3 (pid 64244)] {'acc': 0.8884, 'loss': 0.2857}. # << ACC 0.88
# 2023-10-11 20:20:57.219 [1697055648079786/evaluate/3 (pid 64244)] Results on Spanish reviews:
# 2023-10-11 20:20:57.219 [1697055648079786/evaluate/3 (pid 64244)] {'acc': 0.7380, 'loss': 0.5352}  # <<< ACC 0.73 for spanish
# 2023-10-11 20:20:59.668 [1697055648079786/evaluate/3 (pid 64244)] Task finished successfully.
# 2023-10-11 20:20:59.688 [1697055648079786/end/4 (pid 64509)] Task is starting.
# 2023-10-11 20:21:01.779 [1697055648079786/end/4 (pid 64509)] done! great work!
# 2023-10-11 20:21:02.189 [1697055648079786/end/4 (pid 64509)] Task finished successfully.
# 2023-10-11 20:21:02.190 Done!



# python eval_flow.py run --config configs/eval_dro.json

# Metaflow 2.6.0 executing EvalClassifier for user:gitpod
# Validating your flow...
#     The graph looks good!
# Running pylint...
#     Pylint is happy!
# 2023-10-11 20:53:05.947 Workflow starting (run-id 1697057585941912):
# 2023-10-11 20:53:05.972 [1697057585941912/start/1 (pid 71610)] Task is starting.
# 2023-10-11 20:53:08.632 [1697057585941912/start/1 (pid 71610)] Task finished successfully.
# 2023-10-11 20:53:08.655 [1697057585941912/load_system/2 (pid 71690)] Task is starting.
# 2023-10-11 20:53:10.970 [1697057585941912/load_system/2 (pid 71690)] GPU available: False, used: False
# 2023-10-11 20:53:11.362 [1697057585941912/load_system/2 (pid 71690)] TPU available: False, using: 0 TPU cores
# 2023-10-11 20:53:11.363 [1697057585941912/load_system/2 (pid 71690)] IPU available: False, using: 0 IPUs
# 2023-10-11 20:53:11.363 [1697057585941912/load_system/2 (pid 71690)] HPU available: False, using: 0 HPUs
# 2023-10-11 20:53:11.364 [1697057585941912/load_system/2 (pid 71690)] Task finished successfully.
# 2023-10-11 20:53:11.387 [1697057585941912/evaluate/3 (pid 71750)] Task is starting.
# Testing DataLoader 0: 100%|██████████| 32/32 [00:00<00:00, 64.49it/s]ting: 0it [00:00, ?it/s]
# Testing DataLoader 0: 100%|██████████| 32/32 [00:00<00:00, 62.01it/s]ting: 0it [00:00, ?it/s]
# 2023-10-11 20:53:15.316 [1697057585941912/evaluate/3 (pid 71750)] Results on English reviews:
# 2023-10-11 20:53:15.316 [1697057585941912/evaluate/3 (pid 71750)] {'acc': 0.841, 'loss': 0.415} <<< ACC EN 0.84
# 2023-10-11 20:53:17.774 [1697057585941912/evaluate/3 (pid 71750)] Results on Spanish reviews:
# 2023-10-11 20:53:17.774 [1697057585941912/evaluate/3 (pid 71750)] {'acc': 0.741, 'loss': 0.5412} <<< ACC SP 0.74
# 2023-10-11 20:53:17.776 [1697057585941912/evaluate/3 (pid 71750)] Task finished successfully.
# 2023-10-11 20:53:17.799 [1697057585941912/end/4 (pid 72016)] Task is starting.
# 2023-10-11 20:53:19.766 [1697057585941912/end/4 (pid 72016)] done! great work!
# 2023-10-11 20:53:20.251 [1697057585941912/end/4 (pid 72016)] Task finished successfully.
# 2023-10-11 20:53:20.252 Done!

# English went down from 0.89 to 0.84
# Spanish went up   from 0.738 to 0.741