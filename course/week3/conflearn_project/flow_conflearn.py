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


class TrainIdentifyReview(FlowSpec):
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

    self.next(self.crossval)
  
  @step
  def crossval(self):
    r"""Confidence learning requires cross validation to compute 
    out-of-sample probabilities for every element. Each element
    will appear in a single cross validation split exactly once. 
    """
    # combine training and dev datasets
    X = np.concatenate([
      np.asarray(self.dm.train_dataset.embedding),
      np.asarray(self.dm.dev_dataset.embedding),
      np.asarray(self.dm.test_dataset.embedding),
    ])
    y = np.concatenate([
      np.asarray(self.dm.train_dataset.data.label),
      np.asarray(self.dm.dev_dataset.data.label),
      np.asarray(self.dm.test_dataset.data.label),
    ])

    probs = np.zeros(len(X))  # we will fill this in

    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    kf = KFold(n_splits=3)    # create kfold splits

    for train_index, test_index in kf.split(X):
      probs_ = None
      # ===============================================
      # FILL ME OUT
      # 
      # Fit a new `SentimentClassifierSystem` on the split of 
      # `X` and `y` defined by the current `train_index` and
      # `test_index`. Then, compute predicted probabilities on 
      # the test set. Store these probabilities as a 1-D numpy
      # array `probs_`.
      # 
      # Use `self.config.train.optimizer` to specify any hparams 
      # like `batch_size` or `epochs`.
      #  
      # HINT: `X` and `y` are currently numpy objects. You will 
      # need to convert them to torch tensors prior to training. 
      # You may find the `TensorDataset` class useful. Remember 
      # that `Trainer.fit` and `Trainer.predict` take `DataLoaders`
      # as an input argument.
      # 
      # Our solution is ~15 lines of code.
      # 
      # Pseudocode:
      # --
      # Get train and test slices of X and y.
      # Convert to torch tensors.
      # Create train/test datasets using tensors.
      # Create train/test data loaders from datasets.
      # Create `SentimentClassifierSystem`.
      # Create `Trainer` and call `fit`.
      # Call `predict` on `Trainer` and the test data loader.
      # Convert probabilities back to numpy (make sure 1D).
      # 
      # Types:
      # --
      # probs_: np.array[float] (shape: |test set|)
      # ===============================================

      X_train = torch.from_numpy(X[train_index]).float()
      X_test = torch.from_numpy(X[test_index]).float()
      y_train = torch.from_numpy(y[train_index]).long()
      y_test = torch.from_numpy(y[test_index]).long()

      ds_train = TensorDataset(X_train, y_train)
      ds_test = TensorDataset(X_test, y_test)

      dl_train = DataLoader(ds_train, batch_size=self.config.train.optimizer.batch_size, shuffle=True)
      dl_test = DataLoader(ds_test, batch_size=self.config.train.optimizer.batch_size, shuffle=False)

      system = SentimentClassifierSystem(self.config)
      trainer = Trainer(max_epochs=self.config.train.optimizer.max_epochs) # devices=1, accelerator="gpu" etc
      trainer.fit(system, dl_train)

      probs_ = trainer.predict(system, dataloaders=dl_test, ckpt_path = 'best')
      probs_ = torch.cat(probs_).squeeze(1).numpy()

      assert probs_ is not None, "`probs_` is not defined."
      probs[test_index] = probs_

    # create a single dataframe with all input features
    all_df = pd.concat([
      self.dm.train_dataset.data,
      self.dm.dev_dataset.data,
      self.dm.test_dataset.data,
    ])
    all_df = all_df.reset_index(drop=True)
    # add out-of-sample probabilities to the dataframe
    all_df['prob'] = probs

    # save to excel file
    all_df.to_csv(join(DATA_DIR, 'prob.csv'), index=False)

    self.all_df = all_df
    self.next(self.inspect)

  @step
  def inspect(self):
    r"""Use confidence learning over examples to identify labels that 
    likely have issues with the `cleanlab` tool. 
    """
    prob = np.asarray(self.all_df.prob)
    prob = np.stack([1 - prob, prob]).T
  
    # rank label indices by issues
    ranked_label_issues = None
    
    # =============================
    # FILL ME OUT
    # 
    # Apply confidence learning to labels and out-of-sample
    # predicted probabilities. 
    # 
    # HINT: use cleanlab. See tutorial. 
    # 
    # Our solution is one function call.
    # 
    # Types
    # --
    # ranked_label_issues: List[int]
    # =============================
    ranked_label_issues = find_label_issues(self.all_df.label,
                                            prob,
                                            return_indices_ranked_by='self_confidence'
                                            )
    assert ranked_label_issues is not None, "`ranked_label_issues` not defined."

    # save this to class
    self.issues = ranked_label_issues
    print(f'{len(ranked_label_issues)} label issues found.')

    # overwrite label for all the entries in all_df
    for index in self.issues:
      label = self.all_df.loc[index, 'label']
      # we FLIP the label!
      self.all_df.loc[index, 'label'] = 1 - label

    self.next(self.review)

  @step
  def review(self):
    r"""Format the data quality issues found such that they are ready to be 
    imported into LabelStudio. We expect the following format:

    [
      {
        "data": {
          "text": <review text>
        },
        "predictions": [
          {
            "value": {
              "choices": [
                  "Positive"
              ]
            },
            "from_name": "sentiment",
            "to_name": "text",
            "type": "choices"
          }
        ]
      }
    ]

    See https://labelstud.io/guide/predictions.html#Import-pre-annotations-for-text.and

    You do not need to complete anything in this function. However, look through the 
    code and make sure the operations and output make sense.
    """
    outputs = []
    for index in self.issues:
      row = self.all_df.iloc[index]
      output = {
        'data': {
          'text': str(row.review),
        },
        'predictions': [{
          'result': [
            {
              'value': {
                'choices': [
                  'Positive' if row.label == 1 else 'Negative'
                ]
              },
              'id': f'data-{index}',
              'from_name': 'sentiment',
              'to_name': 'text',
              'type': 'choices',
            },
          ],
        }],
      }
      outputs.append(output)

    # save to file
    preanno_path = join(self.config.review.save_dir, 'pre-annotations.json')
    to_json(outputs, preanno_path)

    self.next(self.retrain_retest)

  @step
  def retrain_retest(self):
    r"""Retrain without reviewing. Let's assume all the labels that 
    confidence learning suggested to flip are indeed erroneous."""
    dm = ReviewDataModule(self.config)
    train_size = len(dm.train_dataset)
    dev_size = len(dm.dev_dataset)

    # ====================================
    # FILL ME OUT
    # 
    # Overwrite the dataframe in each dataset with `all_df`. Make sure to 
    # select the right indices. Since `all_df` contains the corrected labels,
    # training on it will incorporate cleanlab's re-annotations.
    # 
    # Pseudocode:
    # --
    # dm.train_dataset.data = training slice of self.all_df
    # dm.dev_dataset.data = dev slice of self.all_df
    # dm.test_dataset.data = test slice of self.all_df
    # # ====================================
    dm.train_dataset.data = self.all_df.iloc[:train_size]
    dm.dev_dataset.data = self.all_df.iloc[train_size:train_size+dev_size]
    dm.test_dataset.data = self.all_df.iloc[train_size+dev_size:]

    # start from scratch
    system = SentimentClassifierSystem(self.config)
    trainer = Trainer(
      max_epochs = self.config.train.optimizer.max_epochs)

    trainer.fit(system, dm)
    trainer.test(system, dm, ckpt_path = 'best')
    results = system.test_results

    pprint(results)

    log_file = join(Path(__file__).resolve().parent.parent, 
      f'logs', 'post-results.json')

    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save to disk

    self.next(self.end)

  @step
  def end(self):
    """End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python flow_conflearn.py`. To list
  this flow, run `python flow_conflearn.py show`. To execute
  this flow, run `python flow_conflearn.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python flow_conflearn.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python flow_conflearn.py resume`
  
  You can specify a run id as well.
  """
  flow = TrainIdentifyReview()

# first, download the big lfs files
# cd data
# sudo git lfs pull (you will have to provideyour github id and pw)

# python flow_conflearn.py --no-pylint run
# Metaflow 2.6.0 executing TrainIdentifyReview for user:gitpod
# Validating your flow...
#     The graph looks good!
# 2023-10-06 22:37:59.757 Workflow starting (run-id 1696631879753337):
# 2023-10-06 22:37:59.774 [1696631879753337/start/1 (pid 22638)] Task is starting.
# 2023-10-06 22:38:02.571 [1696631879753337/start/1 (pid 22638)] Task finished successfully.
# 2023-10-06 22:38:02.589 [1696631879753337/init_system/2 (pid 22694)] Task is starting.
# 2023-10-06 22:38:05.203 [1696631879753337/init_system/2 (pid 22694)] GPU available: False, used: False
# 2023-10-06 22:38:05.204 [1696631879753337/init_system/2 (pid 22694)] TPU available: False, using: 0 TPU cores
# 2023-10-06 22:38:05.204 [1696631879753337/init_system/2 (pid 22694)] IPU available: False, using: 0 IPUs
# 2023-10-06 22:38:05.204 [1696631879753337/init_system/2 (pid 22694)] HPU available: False, using: 0 HPUs
# 2023-10-06 22:38:26.626 [1696631879753337/init_system/2 (pid 22694)] Task finished successfully.
# 2023-10-06 22:38:26.646 [1696631879753337/train_test/3 (pid 22782)] Task is starting.
# 2023-10-06 22:38:33.829 [1696631879753337/train_test/3 (pid 22782)] 
# Epoch 1:  80%|████████  | 1600/2000 [00:25<00:06, 62.49it/s, loss=0.427, v_num=1, train_loss=0.596, train_acc=0.688, dev_loss=0.528, dev_acc=0.807]
# 2023-10-06 22:39:27.488 [1696631879753337/train_test/3 (pid 22782)] | Name  | Type       | Params|██████████| 400/400 [00:04<00:00, 97.80it/s] 
# 2023-10-06 22:39:27.489 [1696631879753337/train_test/3 (pid 22782)] -------------------------------------
# 2023-10-06 22:39:27.489 [1696631879753337/train_test/3 (pid 22782)] 0 | model | Sequential | 49.3 K400 [00:00<?, ?it/s]
# Epoch 1:  80%|████████  | 1601/2000 [00:25<00:06, 62.51it/s, loss=0.427, v_num=1, train_loss=0.596, train_acc=0.688, dev_loss=0.528, dev_acc=0.807]
# Epoch 1:  80%|████████  | 1602/2000 [00:25<00:06, 62.41it/s, loss=0.427, v_num=1, train_loss=0.596, train_acc=0.688, dev_loss=0.528, dev_acc=0.807]
# Epoch 1:  80%|████████  | 1603/2000 [00:25<00:06, 62.44it/s, loss=0.427, v_num=1, train_loss=0.596, train_acc=0.688, dev_loss=0.528, dev_acc=0.807]
# Epoch 1:  80%|████████  | 1604/2000 [00:25<00:06, 62.46it/s, loss=0.427, v_num=1, train_loss=0.596, train_acc=0.688, dev_loss=0.528, dev_acc=0.807]
# Epoch 1:  80%|████████  | 1605/2000 [00:25<00:06, 62.49it/s, loss=0.427, v_num=1, train_loss=0.596, train_acc=0.688, dev_loss=0.528, dev_acc=0.807]
# Epoch 2:  80%|████████  | 1600/2000 [00:27<00:06, 58.68it/s, loss=0.367, v_num=1, train_loss=0.448, train_acc=0.781, dev_loss=0.405, dev_acc=0.840]del to '/workspace/DCDL/course/week3/conflearn_project/log/epoch=0-step=1600.ckpt' as top 1
# Epoch 3:  80%|████████  | 1600/2000 [00:25<00:06, 62.99it/s, loss=0.352, v_num=1, train_loss=0.532, train_acc=0.781, dev_loss=0.353, dev_acc=0.854]del to '/workspace/DCDL/course/week3/conflearn_project/log/epoch=1-step=3200.ckpt' as top 1
# Epoch 4:  80%|████████  | 1600/2000 [00:25<00:06, 62.29it/s, loss=0.317, v_num=1, train_loss=0.412, train_acc=0.781, dev_loss=0.366, dev_acc=0.833]del to '/workspace/DCDL/course/week3/conflearn_project/log/epoch=2-step=4800.ckpt' as top 1
# Epoch 5:  80%|████████  | 1600/2000 [00:29<00:07, 53.82it/s, loss=0.312, v_num=1, train_loss=0.344, train_acc=0.844, dev_loss=0.360, dev_acc=0.834] 
# Epoch 6:  80%|████████  | 1600/2000 [00:27<00:06, 58.41it/s, loss=0.288, v_num=1, train_loss=0.183, train_acc=0.969, dev_loss=0.308, dev_acc=0.871] 
# Epoch 7:  80%|████████  | 1600/2000 [00:23<00:05, 68.97it/s, loss=0.299, v_num=1, train_loss=0.339, train_acc=0.844, dev_loss=0.292, dev_acc=0.880] el to '/workspace/DCDL/course/week3/conflearn_project/log/epoch=5-step=9600.ckpt' as top 1
# Epoch 8:  80%|████████  | 1600/2000 [00:24<00:06, 65.26it/s, loss=0.311, v_num=1, train_loss=0.576, train_acc=0.812, dev_loss=0.303, dev_acc=0.877] del to '/workspace/DCDL/course/week3/conflearn_project/log/epoch=6-step=11200.ckpt' as top 1
# Epoch 9:  80%|████████  | 1600/2000 [00:24<00:06, 66.12it/s, loss=0.303, v_num=1, train_loss=0.309, train_acc=0.812, dev_loss=0.298, dev_acc=0.876]
# Epoch 9: 100%|██████████| 2000/2000 [00:28<00:00, 70.47it/s, loss=0.303, v_num=1, train_loss=0.309, train_acc=0.812, dev_loss=0.280, dev_acc=0.886]
# 2023-10-06 22:43:35.984 [1696631879753337/train_test/3 (pid 22782)] Epoch 9, global step 16000: 'dev_loss' reached 0.28007 (best 0.28007), saving model to '/workspace/DCDL/course/week3/conflearn_project/log/epoch=9-step=16000-v1.ckpt' as top 1
# 2023-10-06 22:43:35.988 [1696631879753337/train_test/3 (pid 22782)] Restoring states from the checkpoint path at /workspace/DCDL/course/week3/conflearn_project/log/epoch=9-step=16000-v1.ckpt
# 2023-10-06 22:43:35.993 [1696631879753337/train_test/3 (pid 22782)] Loaded model weights from checkpoint at /workspace/DCDL/course/week3/conflearn_project/log/epoch=9-step=16000-v1.ckpt
# Testing DataLoader 0: 100%|██████████| 500/500 [00:04<00:00, 101.68it/s]ing: 0it [00:00, ?it/s]
# 2023-10-06 22:43:40.918 [1696631879753337/train_test/3 (pid 22782)] {'acc': 0.8899999856948853, 'loss': 0.2831442356109619}
# 2023-10-06 22:44:44.170 [1696631879753337/train_test/3 (pid 22782)] Task finished successfully.
# 2023-10-06 22:44:44.188 [1696631879753337/crossval/4 (pid 23580)] Task is starting.
# 2023-10-06 22:44:51.419 [1696631879753337/crossval/4 (pid 23580)] GPU available: False, used: False
# 2023-10-06 22:44:51.419 [1696631879753337/crossval/4 (pid 23580)] TPU available: False, using: 0 TPU cores
# 2023-10-06 22:44:51.419 [1696631879753337/crossval/4 (pid 23580)] IPU available: False, using: 0 IPUs
# 2023-10-06 22:44:51.419 [1696631879753337/crossval/4 (pid 23580)] HPU available: False, using: 0 HPUs
# 2023-10-06 22:44:51.422 [1696631879753337/crossval/4 (pid 23580)] 
# Epoch 9: 100%|██████████| 1667/1667 [00:12<00:00, 135.21it/s, loss=0.343, v_num=2, train_loss=0.533, train_acc=0.810]
# 2023-10-06 22:47:27.123 [1696631879753337/crossval/4 (pid 23580)] | Name  | Type       | Params
# Predicting DataLoader 0: 100%|██████████| 834/834 [00:34<00:00, -24.08it/s]   1667it [00:00, ?it/s]
# 2023-10-06 22:47:27.489 [1696631879753337/crossval/4 (pid 23580)] -------------------------------------
# 2023-10-06 22:47:27.489 [1696631879753337/crossval/4 (pid 23580)] 0 | model | Sequential | 49.3 K
# 2023-10-06 22:47:27.489 [1696631879753337/crossval/4 (pid 23580)] -------------------------------------
# 2023-10-06 22:47:27.489 [1696631879753337/crossval/4 (pid 23580)] 49.3 K    Trainable params
# 2023-10-06 22:47:27.489 [1696631879753337/crossval/4 (pid 23580)] 0         Non-trainable params
# 2023-10-06 22:47:27.489 [1696631879753337/crossval/4 (pid 23580)] 49.3 K    Total params
# 2023-10-06 22:47:27.489 [1696631879753337/crossval/4 (pid 23580)] 0.197     Total estimated model params size (MB)
# 2023-10-06 22:47:27.489 [1696631879753337/crossval/4 (pid 23580)] Restoring states from the checkpoint path at /workspace/DCDL/course/week3/conflearn_project/lightning_logs/version_2/checkpoints/epoch=9-step=16670.ckpt
# 2023-10-06 22:47:27.536 [1696631879753337/crossval/4 (pid 23580)] Loaded model weights from checkpoint at /workspace/DCDL/course/week3/conflearn_project/lightning_logs/version_2/checkpoints/epoch=9-step=16670.ckpt
# 2023-10-06 22:47:27.536 [1696631879753337/crossval/4 (pid 23580)] GPU available: False, used: False
# 2023-10-06 22:47:27.538 [1696631879753337/crossval/4 (pid 23580)] TPU available: False, using: 0 TPU cores
# 2023-10-06 22:47:27.538 [1696631879753337/crossval/4 (pid 23580)] IPU available: False, using: 0 IPUs
# 2023-10-06 22:47:27.538 [1696631879753337/crossval/4 (pid 23580)] HPU available: False, using: 0 HPUs
# 2023-10-06 22:47:27.538 [1696631879753337/crossval/4 (pid 23580)] 
# Epoch 9: 100%|██████████| 1667/1667 [00:12<00:00, 132.36it/s, loss=0.287, v_num=3, train_loss=0.233, train_acc=0.952] 
# 2023-10-06 22:49:32.487 [1696631879753337/crossval/4 (pid 23580)] | Name  | Type       | Params
# Predicting DataLoader 0: 100%|██████████| 834/834 [00:02<00:00, -278.72it/s]  1667it [00:00, ?it/s]
# 2023-10-06 22:49:32.609 [1696631879753337/crossval/4 (pid 23580)] -------------------------------------
# 2023-10-06 22:49:32.609 [1696631879753337/crossval/4 (pid 23580)] 0 | model | Sequential | 49.3 K
# 2023-10-06 22:49:32.609 [1696631879753337/crossval/4 (pid 23580)] -------------------------------------
# 2023-10-06 22:49:32.609 [1696631879753337/crossval/4 (pid 23580)] 49.3 K    Trainable params
# 2023-10-06 22:49:32.610 [1696631879753337/crossval/4 (pid 23580)] 0         Non-trainable params
# 2023-10-06 22:49:32.610 [1696631879753337/crossval/4 (pid 23580)] 49.3 K    Total params
# 2023-10-06 22:49:32.610 [1696631879753337/crossval/4 (pid 23580)] 0.197     Total estimated model params size (MB)
# 2023-10-06 22:49:32.610 [1696631879753337/crossval/4 (pid 23580)] Restoring states from the checkpoint path at /workspace/DCDL/course/week3/conflearn_project/lightning_logs/version_3/checkpoints/epoch=9-step=16670.ckpt
# 2023-10-06 22:49:32.647 [1696631879753337/crossval/4 (pid 23580)] Loaded model weights from checkpoint at /workspace/DCDL/course/week3/conflearn_project/lightning_logs/version_3/checkpoints/epoch=9-step=16670.ckpt
# 2023-10-06 22:49:32.647 [1696631879753337/crossval/4 (pid 23580)] GPU available: False, used: False
# 2023-10-06 22:49:32.650 [1696631879753337/crossval/4 (pid 23580)] TPU available: False, using: 0 TPU cores
# 2023-10-06 22:49:32.650 [1696631879753337/crossval/4 (pid 23580)] IPU available: False, using: 0 IPUs
# 2023-10-06 22:49:32.650 [1696631879753337/crossval/4 (pid 23580)] HPU available: False, using: 0 HPUs
# 2023-10-06 22:49:32.650 [1696631879753337/crossval/4 (pid 23580)] 
# Epoch 9: 100%|██████████| 1667/1667 [00:11<00:00, 143.48it/s, loss=0.309, v_num=4, train_loss=0.311, train_acc=0.864]
# 2023-10-06 22:51:35.544 [1696631879753337/crossval/4 (pid 23580)] | Name  | Type       | Params
# Predicting DataLoader 0: 100%|██████████| 834/834 [00:02<00:00, -286.66it/s]  1667it [00:00, ?it/s]
# 2023-10-06 22:51:35.626 [1696631879753337/crossval/4 (pid 23580)] -------------------------------------
# 2023-10-06 22:51:35.626 [1696631879753337/crossval/4 (pid 23580)] 0 | model | Sequential | 49.3 K
# 2023-10-06 22:51:35.626 [1696631879753337/crossval/4 (pid 23580)] -------------------------------------
# 2023-10-06 22:51:35.626 [1696631879753337/crossval/4 (pid 23580)] 49.3 K    Trainable params
# 2023-10-06 22:51:35.626 [1696631879753337/crossval/4 (pid 23580)] 0         Non-trainable params
# 2023-10-06 22:51:35.626 [1696631879753337/crossval/4 (pid 23580)] 49.3 K    Total params
# 2023-10-06 22:51:35.626 [1696631879753337/crossval/4 (pid 23580)] 0.197     Total estimated model params size (MB)
# 2023-10-06 22:51:35.627 [1696631879753337/crossval/4 (pid 23580)] Restoring states from the checkpoint path at /workspace/DCDL/course/week3/conflearn_project/lightning_logs/version_4/checkpoints/epoch=9-step=16670.ckpt
# 2023-10-06 22:51:58.088 [1696631879753337/crossval/4 (pid 23580)] Loaded model weights from checkpoint at /workspace/DCDL/course/week3/conflearn_project/lightning_logs/version_4/checkpoints/epoch=9-step=16670.ckpt
# 2023-10-06 22:51:58.090 [1696631879753337/crossval/4 (pid 23580)] Task finished successfully.
# 2023-10-06 22:51:58.116 [1696631879753337/inspect/5 (pid 24108)] Task is starting.
# 2023-10-06 22:52:01.252 [1696631879753337/inspect/5 (pid 24108)] 3951 label issues found.
# 2023-10-06 22:52:03.526 [1696631879753337/inspect/5 (pid 24108)] Task finished successfully.
# 2023-10-06 22:52:03.541 [1696631879753337/review/6 (pid 24194)] Task is starting.
# 2023-10-06 22:52:06.984 [1696631879753337/review/6 (pid 24194)] Task finished successfully.
# 2023-10-06 22:52:06.994 [1696631879753337/retrain_retest/7 (pid 24255)] Task is starting.
# 2023-10-06 22:52:09.673 [1696631879753337/retrain_retest/7 (pid 24255)] GPU available: False, used: False
# 2023-10-06 22:52:09.674 [1696631879753337/retrain_retest/7 (pid 24255)] TPU available: False, using: 0 TPU cores
# 2023-10-06 22:52:09.674 [1696631879753337/retrain_retest/7 (pid 24255)] IPU available: False, using: 0 IPUs
# 2023-10-06 22:52:09.677 [1696631879753337/retrain_retest/7 (pid 24255)] HPU available: False, using: 0 HPUs
# 2023-10-06 22:52:09.677 [1696631879753337/retrain_retest/7 (pid 24255)] 
# Epoch 5:  80%|████████  | 1600/2000 [00:24<00:06, 64.80it/s, loss=0.264, v_num=5, train_loss=0.314, train_acc=0.844, dev_loss=0.308, dev_acc=0.890]
# 2023-10-06 22:54:59.585 1 tasks are running: e.g. ....st/7 (pid 24255)] Validation DataLoader 0: 100%|██████████| 400/400 [00:04<00:00, 74.34it/s]
# 2023-10-06 22:54:59.585 0 tasks are waiting in the queue.
# Epoch 9: 100%|██████████| 2000/2000 [00:29<00:00, 68.04it/s, loss=0.225, v_num=5, train_loss=0.185, train_acc=0.906, dev_loss=0.230, dev_acc=0.910] 
# 2023-10-06 22:57:02.411 [1696631879753337/retrain_retest/7 (pid 24255)] | Name  | Type       | Params|██████████| 400/400 [00:04<00:00, 83.36it/s]
# 2023-10-06 22:57:02.411 [1696631879753337/retrain_retest/7 (pid 24255)] -------------------------------------
# 2023-10-06 22:57:02.411 [1696631879753337/retrain_retest/7 (pid 24255)] 0 | model | Sequential | 49.3 K
# 2023-10-06 22:57:02.411 [1696631879753337/retrain_retest/7 (pid 24255)] -------------------------------------
# 2023-10-06 22:57:02.411 [1696631879753337/retrain_retest/7 (pid 24255)] 49.3 K    Trainable params
# 2023-10-06 22:57:02.411 [1696631879753337/retrain_retest/7 (pid 24255)] 0         Non-trainable params
# 2023-10-06 22:57:02.411 [1696631879753337/retrain_retest/7 (pid 24255)] 49.3 K    Total params
# 2023-10-06 22:57:02.411 [1696631879753337/retrain_retest/7 (pid 24255)] 0.197     Total estimated model params size (MB)
# 2023-10-06 22:57:02.412 [1696631879753337/retrain_retest/7 (pid 24255)] Restoring states from the checkpoint path at /workspace/DCDL/course/week3/conflearn_project/lightning_logs/version_5/checkpoints/epoch=9-step=16000.ckpt
# 2023-10-06 22:57:02.416 [1696631879753337/retrain_retest/7 (pid 24255)] Loaded model weights from checkpoint at /workspace/DCDL/course/week3/conflearn_project/lightning_logs/version_5/checkpoints/epoch=9-step=16000.ckpt
# Testing DataLoader 0: 100%|██████████| 500/500 [00:05<00:00, 98.60it/s] Testing: 0it [00:00, ?it/s]
# 2023-10-06 22:57:07.498 [1696631879753337/retrain_retest/7 (pid 24255)] {'acc': 0.9160624742507935, 'loss': 0.22987355291843414}.  <<<< IMPROVED ACC 0.92
# 2023-10-06 22:57:07.924 [1696631879753337/retrain_retest/7 (pid 24255)] Task finished successfully.
# 2023-10-06 22:57:07.940 [1696631879753337/end/8 (pid 24863)] Task is starting.
# 2023-10-06 22:57:10.115 [1696631879753337/end/8 (pid 24863)] done! great work!
# 2023-10-06 22:57:10.460 [1696631879753337/end/8 (pid 24863)] Task finished successfully.
# 2023-10-06 22:57:10.461 Done!