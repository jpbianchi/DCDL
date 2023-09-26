"""
Flow #2: This flow will train a multilayer perceptron with a 
width found by hyperparameter search.
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
  digit, such as those in the MNIST dataset. We search over three 
  potential different widths.

  Arguments
  ---------
  config (str, default: ./config.py): path to a configuration file
  """ 
  config_path = Parameter('config', 
    help = 'path to config file', 
    default='./configs/hparam_flow.json')

  @step
  def start(self):
    r"""Start node.
    Set random seeds for reproducibility.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    self.widths = [16, 32, 64]

    # for each width, we will train a model
    self.next(self.init_and_train, foreach='widths')

  @step
  def init_and_train(self):
    r"""Instantiates a data module, pytorch lightning module, 
    and lightning trainer instance. Calls `fit` on the trainer.

    This should remind of you of `init_system` and `train_model`
    from Flow #1. We merge the two into one node.
    """
    config = load_config(self.config_path)
    config.system.model.width = self.input

    dm = MNISTDataModule(config)
    system = DigitClassifierSystem(config)

    checkpoint_callback = ModelCheckpoint(
      dirpath = os.path.join(config.system.save_dir, f'width{self.input}'),
      monitor = 'dev_loss',
      mode = 'min',
      save_top_k = 1,
      verbose = True,
    )

    trainer = Trainer(
      max_epochs = config.system.optimizer.max_epochs,
      callbacks = [checkpoint_callback])

    trainer.fit(system, dm)

    self.dm = dm
    self.system = system
    self.trainer = trainer
    self.callback = checkpoint_callback

    self.next(self.find_best)

  @step
  def find_best(self, inputs):
    r"""Only keep the system with the lowest `dev_loss."""

    # manually propagate class variables through but we only
    # need a few of them so no need to call `merge_artifacts`
    self.dm = inputs[0].dm  # dm = data module 

    scores = []        # populate with scores from each hparams
    best_index = None  # replace with best index
    
    # ================================
    # FILL ME OUT
    # 
    # Aggregate the best validation performance across inputs into
    # the variable `scores`.
    # 
    # HINT: the `callback` object has a property `best_model_score`
    #       that make come in handy. 
    # 
    # Then, compute the index of the model and store it in `best_index`.
    # 
    # Pseudocode:
    # --
    # aggregate scores using `inputs`
    # best_index = ...
    scores = [inp.callback.best_model_score for inp in inputs]
    best_index = np.argmin(scores)
    #
    # Type:
    # --
    # scores: List[float] 
    # best_index: integer 
    # ================================

    # sanity check for scores length
    assert len(scores) == len(list(inputs)), "Hmm. Incorrect length for scores."
    # sanity check for best_index
    assert best_index is not None
    assert best_index >= 0 and best_index < len(list(inputs))
    
    # get the best system / trainer
    # we drop the callback
    self.system = inputs[best_index].system
    self.trainer = inputs[best_index].trainer

    # save the best width
    self.best_width = inputs[best_index].widths[best_index]

    # only the best model proceeds to offline test
    self.next(self.offline_test)

  @step
  def offline_test(self):
    r"""Calls (offline) `test` on the trainer. Saves results to a log file."""

    self.trainer.test(self.system, self.dm, ckpt_path = 'best')
    results = self.system.test_results
    results['best_width'] = self.best_width

    pprint(results)

    log_file = join(Path(__file__).resolve().parent.parent, 
      f'logs/hparam_flow', 'offline-test-results.json')

    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save to disk

    self.next(self.end)  

  @step
  def end(self):
    """End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python hparam_flow.py`. To list
  this flow, run `python hparam_flow.py show`. To execute
  this flow, run `python hparam_flow.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python hparam_flow.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python hparam_flow.py resume`
  
  You can specify a run id as well.
  """
  flow = DigitClassifierFlow()


# gitpod /workspace/data-centric-deep-learning/course/week2/pipeline_project (main) $ python flows/hparam_flow.py run
# Metaflow 2.6.0 executing DigitClassifierFlow for user:gitpod
# Validating your flow...
#     The graph looks good!
# Running pylint...
#     Pylint is happy!
# 2023-09-26 21:54:39.122 Workflow starting (run-id 1695765279117523):
# 2023-09-26 21:54:39.144 [1695765279117523/start/1 (pid 24889)] Task is starting.
# 2023-09-26 21:54:41.215 [1695765279117523/start/1 (pid 24889)] Foreach yields 3 child steps.
# 2023-09-26 21:54:41.215 [1695765279117523/start/1 (pid 24889)] Task finished successfully.
# 2023-09-26 21:54:41.237 [1695765279117523/init_and_train/2 (pid 24923)] Task is starting.
# 2023-09-26 21:54:41.255 [1695765279117523/init_and_train/3 (pid 24924)] Task is starting.
# 2023-09-26 21:54:41.273 [1695765279117523/init_and_train/4 (pid 24925)] Task is starting.
# 2023-09-26 21:54:43.342 [1695765279117523/init_and_train/2 (pid 24923)] GPU available: False, used: False
# 2023-09-26 21:54:43.343 [1695765279117523/init_and_train/2 (pid 24923)] TPU available: False, using: 0 TPU cores
# 2023-09-26 21:54:43.343 [1695765279117523/init_and_train/2 (pid 24923)] IPU available: False, using: 0 IPUs
# 2023-09-26 21:54:43.343 [1695765279117523/init_and_train/2 (pid 24923)] HPU available: False, using: 0 HPUs
# 2023-09-26 21:54:43.345 [1695765279117523/init_and_train/2 (pid 24923)] 
# Epoch 0:  80%|████████  | 1500/1875 [00:24<00:06, 60.40it/s, loss=0.209, v_num=1, train_loss=0.228, train_acc=0.938] ]
# 2023-09-26 21:55:08.271 [1695765279117523/init_and_train/3 (pid 24924)] GPU available: False, used: False
# Epoch 0:  80%|████████  | 1500/1875 [00:37<00:09, 39.48it/s, loss=0.21, v_num=2, train_loss=0.271, train_acc=0.906]] 
# 2023-09-26 21:55:21.542 [1695765279117523/init_and_train/4 (pid 24925)] GPU available: False, used: False
# Epoch 0:  85%|████████▌ | 1600/1875 [01:05<00:11, 24.36it/s, loss=0.209, v_num=3, train_loss=0.159, train_acc=0.938]]9]
# Epoch 0: 100%|█████████▉| 1874/1875 [01:06<00:00, 28.35it/s, loss=0.209, v_num=1, train_loss=0.228, train_acc=0.938]0/375 [00:07<00:19, 13.95it/s]
# Epoch 0:  98%|█████████▊| 1843/1875 [01:05<00:01, 27.97it/s, loss=0.21, v_num=2, train_loss=0.271, train_acc=0.906]01/375 [00:07<00:25, 10.96it/s]
# 2023-09-26 21:55:49.810 [1695765279117523/init_and_train/2 (pid 24923)] 0 | model | Sequential | 12.7 K████████▏| 343/375 [00:27<00:02, 11.02it/s]
# 2023-09-26 21:55:49.810 [1695765279117523/init_and_train/2 (pid 24923)] -------------------------------------
# 2023-09-26 21:55:49.810 [1695765279117523/init_and_train/2 (pid 24923)] 12.7 K    Trainable params
# 2023-09-26 21:55:49.811 [1695765279117523/init_and_train/2 (pid 24923)] 0         Non-trainable params
# 2023-09-26 21:55:49.811 [1695765279117523/init_and_train/2 (pid 24923)] 12.7 K    Total params
# 2023-09-26 21:55:49.811 [1695765279117523/init_and_train/2 (pid 24923)] 0.051     Total estimated model params size (MB)
# 2023-09-26 21:55:49.811 [1695765279117523/init_and_train/2 (pid 24923)] Epoch 0, global step 1500: 'dev_loss' reached 0.24003 (best 0.24003), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/hpEpoch 0:  99%|█████████▊| 1851/1875 [01:06<00:00, 27.84it/s, loss=0.21, v_num=2, train_loss=0.271, train_acc=0.906]9, dev_loss=0.240, dev_acc=0.928]]
# Epoch 0:  99%|█████████▉| 1852/1875 [01:06<00:00, 27.85it/s, loss=0.21, v_num=2, train_loss=0.271, train_acc=0.906]51/375 [00:28<00:01, 13.47it/s]
# Epoch 0:  86%|████████▌ | 1605/1875 [01:06<00:11, 24.24it/s, loss=0.209, v_num=3, train_loss=0.159, train_acc=0.938]5/375 [00:07<00:22, 11.90it/s]
# Epoch 0:  86%|████████▌ | 1606/1875 [01:06<00:11, 24.23it/s, loss=0.209, v_num=3, train_loss=0.159, train_acc=0.938]5/375 [00:07<00:22, 11.90it/s]
# Epoch 0:  86%|████████▌ | 1607/1875 [01:06<00:11, 24.24it/s, loss=0.209, v_num=3, train_loss=0.159, train_acc=0.938] ?it/s]00:08<00:22, 11.90it/s]
# Epoch 1:  80%|████████  | 1501/1875 [00:19<00:04, 75.47it/s, loss=0.196, v_num=1, train_loss=0.0524, train_acc=0.969, dev_loss=0.240, dev_acc=0.928]
# Epoch 1:  80%|████████  | 1502/1875 [00:19<00:04, 75.50it/s, loss=0.196, v_num=1, train_loss=0.0524, train_acc=0.969, dev_loss=0.240, dev_acc=0.928]
# Epoch 1:  80%|████████  | 1503/1875 [00:19<00:04, 75.54it/s, loss=0.196, v_num=1, train_loss=0.0524, train_acc=0.969, dev_loss=0.240, dev_acc=0.928]
# Epoch 1:  80%|████████  | 1504/1875 [00:19<00:04, 75.57it/s, loss=0.196, v_num=1, train_loss=0.0524, train_acc=0.969, dev_loss=0.240, dev_acc=0.928]
# Epoch 1:  80%|████████  | 1505/1875 [00:19<00:04, 75.61it/s, loss=0.196, v_num=1, train_loss=0.0524, train_acc=0.969, dev_loss=0.240, dev_acc=0.928]
# Epoch 1:  80%|████████  | 1506/1875 [00:19<00:04, 75.64it/s, loss=0.196, v_num=1, train_loss=0.0524, train_acc=0.969, dev_loss=0.240, dev_acc=0.928]
# Epoch 0:  99%|█████████▉| 1860/1875 [01:06<00:00, 27.85it/s, loss=0.21, v_num=2, train_loss=0.271, train_acc=0.906]/375 [00:00<00:01, 275.86it/s]]
# Epoch 0:  86%|████████▌ | 1613/1875 [01:06<00:10, 24.22it/s, loss=0.209, v_num=3, train_loss=0.159, train_acc=0.938](MB)5 [00:08<00:14, 17.73it/s]
# 2023-09-26 21:56:09.643 [1695765279117523/init_and_train/3 (pid 24924)] Epoch 0, global step 1500: 'dev_loss' reached 0.18382 (best 0.18382), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/hpEpoch 1:  80%|████████  | 1507/1875 [00:19<00:04, 75.67it/s, loss=0.196, v_num=1, train_loss=0.0524, train_acc=0.969, dev_loss=0.240, dev_acc=0.928]7]]
# 2023-09-26 21:56:24.014 [1695765279117523/init_and_train/4 (pid 24925)] TPU available: False, using: 0 TPU cores| 7/375 [00:00<00:01, 244.73it/s]]
# Epoch 1:  80%|████████  | 1508/1875 [00:20<00:04, 75.39it/s, loss=0.196, v_num=1, train_loss=0.0524, train_acc=0.969, dev_loss=0.240, dev_acc=0.928]
# Epoch 1:  80%|████████  | 1509/1875 [00:20<00:04, 75.42it/s, loss=0.196, v_num=1, train_loss=0.0524, train_acc=0.969, dev_loss=0.240, dev_acc=0.928]
# Epoch 1:  81%|████████  | 1510/1875 [00:20<00:04, 75.17it/s, loss=0.196, v_num=1, train_loss=0.0524, train_acc=0.969, dev_loss=0.240, dev_acc=0.928]
# Epoch 1:  81%|████████  | 1511/1875 [00:20<00:04, 75.20it/s, loss=0.196, v_num=1, train_loss=0.0524, train_acc=0.969, dev_loss=0.240, dev_acc=0.928]
# Epoch 1:  81%|████████  | 1512/1875 [00:20<00:04, 75.23it/s, loss=0.196, v_num=1, train_loss=0.0524, train_acc=0.969, dev_loss=0.240, dev_acc=0.928]
# Epoch 1:  80%|████████  | 1501/1875 [00:32<00:08, 46.09it/s, loss=0.177, v_num=2, train_loss=0.0968, train_acc=0.969, dev_loss=0.184, dev_acc=0.947]
# Epoch 1:  80%|████████  | 1502/1875 [00:32<00:08, 46.08it/s, loss=0.177, v_num=2, train_loss=0.0968, train_acc=0.969, dev_loss=0.184, dev_acc=0.947]
# Epoch 1:  81%|████████  | 1515/1875 [00:20<00:04, 74.24it/s, loss=0.196, v_num=1, train_loss=0.0524, train_acc=0.969, dev_loss=0.240, dev_acc=0.928]
# Epoch 0:  87%|████████▋ | 1637/1875 [01:08<00:09, 24.01it/s, loss=0.209, v_num=3, train_loss=0.159, train_acc=0.938]/375 [00:00<00:05, 70.05it/s]]
# Epoch 1:  81%|████████  | 1516/1875 [00:20<00:04, 73.57it/s, loss=0.196, v_num=1, train_loss=0.0524, train_acc=0.969, dev_loss=0.240, dev_acc=0.928]
# Epoch 1:  81%|████████  | 1517/1875 [00:20<00:04, 72.92it/s, loss=0.196, v_num=1, train_loss=0.0524, train_acc=0.969, dev_loss=0.240, dev_acc=0.928]
# 2023-09-26 21:56:24.019 [1695765279117523/init_and_train/4 (pid 24925)] Epoch 0, global step 1500: 'dev_loss' reached 0.15875 (best 0.15875), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/hpEpoch 1:  80%|████████  | 1500/1875 [00:47<00:11, 31.76it/s, loss=0.135, v_num=3, train_loss=0.120, train_acc=0.969, dev_loss=0.159, dev_acc=0.952]]  
# 2023-09-26 21:56:58.120 [1695765279117523/init_and_train/2 (pid 24923)] Epoch 1, global step 3000: 'dev_loss' reached 0.21998 (best 0.21998), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/hpEpoch 2:  83%|████████▎ | 1554/1875 [00:45<00:09, 34.30it/s, loss=0.211, v_num=1, train_loss=0.173, train_acc=0.906, dev_loss=0.220, dev_acc=0.935]]] 
# 2023-09-26 21:57:18.117 [1695765279117523/init_and_train/3 (pid 24924)] Epoch 1, global step 3000: 'dev_loss' reached 0.16067 (best 0.16067), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/hpEpoch 1:  94%|█████████▍| 1764/1875 [01:07<00:04, 26.25it/s, loss=0.135, v_num=3, train_loss=0.120, train_acc=0.969, dev_loss=0.159, dev_acc=0.952]  
# 2023-09-26 21:57:42.659 [1695765279117523/init_and_train/4 (pid 24925)] Epoch 1, global step 3000: 'dev_loss' reached 0.13924 (best 0.13924), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/hpEpoch 2:  82%|████████▏ | 1530/1875 [00:26<00:05, 58.62it/s, loss=0.13, v_num=2, train_loss=0.106, train_acc=0.969, dev_loss=0.161, dev_acc=0.952]]   ]
# 2023-09-26 21:58:11.322 [1695765279117523/init_and_train/2 (pid 24923)] Epoch 2, global step 4500: 'dev_loss' reached 0.20341 (best 0.20341), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/hpEpoch 3:  80%|████████  | 1500/1875 [00:20<00:05, 74.33it/s, loss=0.141, v_num=1, train_loss=0.0455, train_acc=1.000, dev_loss=0.203, dev_acc=0.941]]
# Epoch 3:  80%|████████  | 1508/1875 [00:32<00:08, 45.85it/s, loss=0.123, v_num=2, train_loss=0.095, train_acc=0.938, dev_loss=0.163, dev_acc=0.953]]] ]
# 2023-09-26 21:58:52.011 [1695765279117523/init_and_train/4 (pid 24925)] Epoch 2, global step 4500: 'dev_loss' reached 0.13449 (best 0.13449), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/hpEpoch 3:  80%|████████  | 1500/1875 [00:34<00:08, 43.49it/s, loss=0.0794, v_num=3, train_loss=0.0995, train_acc=0.969, dev_loss=0.134, dev_acc=0.961]  
# Epoch 3:  97%|█████████▋| 1812/1875 [00:54<00:01, 33.02it/s, loss=0.0794, v_num=3, train_loss=0.0995, train_acc=0.969, dev_loss=0.134, dev_acc=0.961]]
# 2023-09-26 21:59:47.011 [1695765279117523/init_and_train/3 (pid 24924)] Epoch 3, global step 6000: 'dev_loss' reached 0.14847 (best 0.14847), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/hpEpoch 4:  83%|████████▎ | 1564/1875 [00:46<00:09, 33.35it/s, loss=0.165, v_num=1, train_loss=0.150, train_acc=0.938, dev_loss=0.223, dev_acc=0.936]]]7]
# Epoch 4:  81%|████████  | 1511/1875 [00:45<00:10, 33.13it/s, loss=0.0708, v_num=3, train_loss=0.00888, train_acc=1.000, dev_loss=0.138, dev_acc=0.962] 
# Epoch 5:  80%|████████  | 1500/1875 [00:19<00:04, 75.49it/s, loss=0.181, v_num=1, train_loss=0.151, train_acc=0.969, dev_loss=0.213, dev_acc=0.939] 9]
# 2023-09-26 22:00:56.716 [1695765279117523/init_and_train/3 (pid 24924)] Epoch 4, global step 7500: 'dev_loss' reached 0.14530 (best 0.14530), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/hpEpoch 5:  80%|████████  | 1500/1875 [00:33<00:08, 44.50it/s, loss=0.0611, v_num=3, train_loss=0.0013, train_acc=1.000, dev_loss=0.146, dev_acc=0.963]   
# Epoch 5:  82%|████████▏ | 1537/1875 [00:33<00:07, 45.89it/s, loss=0.0929, v_num=2, train_loss=0.00358, train_acc=1.000, dev_loss=0.145, dev_acc=0.960]
# Epoch 5:  95%|█████████▌| 1790/1875 [00:50<00:02, 35.11it/s, loss=0.0611, v_num=3, train_loss=0.0013, train_acc=1.000, dev_loss=0.146, dev_acc=0.963]]]
# Epoch 5:  96%|█████████▌| 1797/1875 [00:51<00:02, 34.89it/s, loss=0.0611, v_num=3, train_loss=0.0013, train_acc=1.000, dev_loss=0.146, dev_acc=0.963] ]] 
# Epoch 7:  80%|████████  | 1500/1875 [00:17<00:04, 83.45it/s, loss=0.128, v_num=1, train_loss=0.128, train_acc=0.938, dev_loss=0.189, dev_acc=0.943] ] 3]
# 2023-09-26 22:03:17.718 [1695765279117523/init_and_train/2 (pid 24923)] Epoch 6, global step 10500: 'dev_loss' reached 0.18896 (best 0.18896), saving model to '/workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/hparam_flow/width16/epoch=6-step=10500.ckpt' as top 1
# Epoch 7:  96%|█████████▌| 1800/1875 [00:38<00:01, 47.00it/s, loss=0.128, v_num=1, train_loss=0.128, train_acc=0.938, dev_loss=0.189, dev_acc=0.943]3] ]
# Epoch 7:  80%|████████  | 1500/1875 [00:32<00:08, 46.60it/s, loss=0.0618, v_num=3, train_loss=0.157, train_acc=0.969, dev_loss=0.154, dev_acc=0.964]4]4]
# Epoch 8:  82%|████████▏ | 1533/1875 [00:44<00:09, 34.84it/s, loss=0.149, v_num=1, train_loss=0.0892, train_acc=0.969, dev_loss=0.190, dev_acc=0.945]6] 
# 2023-09-26 22:04:51.756 [1695765279117523/init_and_train/3 (pid 24924)] Epoch 7, global step 12000: 'dev_loss' was not in top 1<00:27, 12.41it/s]]
# Epoch 9:  80%|████████  | 1500/1875 [00:19<00:04, 76.69it/s, loss=0.143, v_num=1, train_loss=0.0463, train_acc=1.000, dev_loss=0.197, dev_acc=0.946]   ]
# 2023-09-26 22:05:39.293 [1695765279117523/init_and_train/2 (pid 24923)] Epoch 8, global step 13500: 'dev_loss' was not in top 17<00:07, 13.73it/s]
# Epoch 9:  95%|█████████▍| 1780/1875 [00:41<00:02, 42.69it/s, loss=0.143, v_num=1, train_loss=0.0463, train_acc=1.000, dev_loss=0.197, dev_acc=0.946]]  
# Epoch 9:  80%|████████  | 1500/1875 [00:30<00:07, 48.48it/s, loss=0.0459, v_num=3, train_loss=0.0211, train_acc=1.000, dev_loss=0.168, dev_acc=0.966]  ] 
# Epoch 9: 100%|██████████| 1875/1875 [00:48<00:00, 39.05it/s, loss=0.143, v_num=1, train_loss=0.0463, train_acc=1.000, dev_loss=0.214, dev_acc=0.938]
# Epoch 9: 100%|██████████| 1875/1875 [01:11<00:00, 26.26it/s, loss=0.0953, v_num=2, train_loss=0.0605, train_acc=0.969, dev_loss=0.183, dev_acc=0.957]3]
# Epoch 9: 100%|██████████| 1875/1875 [00:36<00:00, 51.77it/s, loss=0.0459, v_num=3, train_loss=0.0211, train_acc=1.000, dev_loss=0.158, dev_acc=0.968]
# 2023-09-26 22:06:37.533 [1695765279117523/init_and_train/4 (pid 24925)] Epoch 9, global step 15000: 'dev_loss' was not in top 15<00:00, 106.37it/s]
# 2023-09-26 22:06:42.243 [1695765279117523/init_and_train/3 (pid 24924)] Task finished successfully.
# 2023-09-26 22:06:43.428 [1695765279117523/init_and_train/4 (pid 24925)] Task finished successfully.
# 2023-09-26 22:06:43.462 [1695765279117523/find_best/5 (pid 176943)] Task is starting.
# 2023-09-26 22:06:52.672 [1695765279117523/find_best/5 (pid 176943)] Task finished successfully.
# 2023-09-26 22:06:52.696 [1695765279117523/offline_test/6 (pid 176989)] Task is starting.
# 2023-09-26 22:06:56.047 [1695765279117523/offline_test/6 (pid 176989)] Restoring states from the checkpoint path at /workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/hparam_flow/width64/epoch=2-step=4500.ckpt
# 2023-09-26 22:06:56.065 [1695765279117523/offline_test/6 (pid 176989)] Loaded model weights from checkpoint at /workspace/data-centric-deep-learning/course/week2/pipeline_project/artifacts/ckpts/hparam_flow/width64/epoch=2-step=4500.ckpt
# Testing DataLoader 0: 100%|██████████| 313/313 [00:02<00:00, 127.10it/s]s]ting: 0it [00:00, ?it/s]
# 2023-09-26 22:06:58.536 [1695765279117523/offline_test/6 (pid 176989)] {'acc': 0.9633586406707764, 'best_width': 64, 'loss': 0.12444242089986801}  <<<<<< RESULT! acc 0.96 now
# 2023-09-26 22:07:03.032 [1695765279117523/offline_test/6 (pid 176989)] Task finished successfully.
# 2023-09-26 22:07:03.057 [1695765279117523/end/7 (pid 177050)] Task is starting.
# 2023-09-26 22:07:04.858 [1695765279117523/end/7 (pid 177050)] done! great work!
# 2023-09-26 22:07:05.121 [1695765279117523/end/7 (pid 177050)] Task finished successfully.
# 2023-09-26 22:07:05.122 Done!