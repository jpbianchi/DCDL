import time
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

from celery import Task, Celery

from src.system import DigitClassifierSystem
from consts import MODEL_PATH, BROKER_URL, BACKEND_URL

# This creates a Celery instance: we specify it to look at the 
# `tasks.py` file and to use Redis as the broker and backend.
app = Celery('tasks', broker=BROKER_URL, backend=BACKEND_URL)


class PredictionTask(Task):
  r"""Celery task to load a pretrained PyTorch Lightning system."""
  abstract = True

  def __init__(self):
    super().__init__()
    self.system = None

  def __call__(self, *args, **kwargs):
    r"""Load lightning system on first call. This way we do not need to 
    load the system on every task request, which quickly gets expensive.
    """
    if self.system is None:
      print(f'Loading digit classifier system from {MODEL_PATH}')
      self.system = self.get_system()
      print('Loading successful.')

    # pass arguments through 
    return self.run(*args, **kwargs)
  
  def get_system(self):
    system = None
    # ================================
    # FILL ME OUT
    # 
    # Load the checkpoint in `MODEL_PATH` using the class 
    # `DigitClassifierSystem`. Store in the variable `system`.
    # 
    # Pseudocode:
    # --
    system = DigitClassifierSystem.load_from_checkpoint(MODEL_PATH)
    # 
    # Types:
    # --
    # system: DigitClassifierSystem
    # ================================
    assert system is not None, "System is not loaded."
    return system.eval()


@app.task(ignore_result=False,
          bind=True,
          base=PredictionTask)
def predict_single(self, data):
  r"""Defines what `PredictionTask.run` should do.
  
  In this case, it will use the loaded LightningSystem to compute
  the forward pass and make a prediction.

  Argument
  --------
  data (str): url denoting image path to do prediction for.

  Returns
  -------
  results (dict[str, any]): response dictionary.
    probs (list[float]): list of probabilities for each MNIST class.
    label (int): predicted class (one with highest probability).
  """
  # image I/O can be very expensive. Put this in worker so we don't 
  # stall on it in FastAPI
  im = Image.open(data)
  im: Image = im.convert('L')  # convert to grayscale
  im_transforms = transforms.Compose([
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
  ])
  im = im_transforms(im)
  im = im.unsqueeze(0)

  # default (placeholder) values
  results = {'label': None, 'probs': None}

  with torch.no_grad():
    logits = None
    # ================================
    # FILL ME OUT
    # 
    # Copy over your solution from `week3_fastapi/api.py`.
    # 
    # Pseudocode:
    # --
    # logits = ... (use system)
    # 
    # Types:
    # --
    # logits: torch.Tensor (shape: 1x10)
    # ================================
    logits = self.system.predict_step(im) # base class is PredictionTask!!
    assert logits is not None, "logits is not defined."

    # To extract the label, just find the largest logit.
    label = torch.argmax(logits, dim=1)  # shape (1)
    label = label.item()                 # tensor -> integer

    probs = None
    # ================================
    # FILL ME OUT
    # 
    # Copy over your solution from `week3_fastapi/api.py`.
    # 
    # Pseudocode:
    # --
    # probs = ...do something to logits...
    # 
    # Types:
    # --
    # probs: torch.Tensor (shape: 1x10)
    # ================================
    probs = F.softmax(logits, dim=-1)
    assert probs is not None, "probs is not defined."
    probs = probs.squeeze(0)        # squeeze to (10) shape
    probs = probs.numpy().tolist()  # convert tensor to list

  results['probs'] = probs
  results['label'] = label

  # ================================
  # NOTE: simulate hard computation! This will help motivate 
  # why we need Celery.
  # 
  # Uncomment when running bash try_api_many_post.sh 
  time.sleep(5)
  # ================================

  return results


# bash try_api_post.sh 
# {"task_id":"70d49437-598d-4dfe-a898-1784b59ddb3e","status":"processing"}gitpod /

# in the celery terminal, we get
# [2023-10-05 21:41:30,035: INFO/MainProcess] Connected to redis://localhost:6379/0
# [2023-10-05 21:41:30,037: INFO/MainProcess] mingle: searching for neighbors
# [2023-10-05 21:41:31,045: INFO/MainProcess] mingle: all alone
# [2023-10-05 21:41:31,059: INFO/MainProcess] celery@jpbianchi-dcdl-uenra6lkt0u ready.
# [2023-10-05 21:42:01,823: INFO/MainProcess] Task tasks.predict_single[e53aa71f-c123-4c29-ad65-0cc3d4dc329e] received
# [2023-10-05 21:42:01,825: WARNING/ForkPoolWorker-14] Loading digit classifier system from /workspace/DCDL/course/week3/celery_project/ckpts/deploy.ckpt
# [2023-10-05 21:42:01,837: INFO/ForkPoolWorker-14] Created a temporary directory at /tmp/tmp0otj9k4r
# [2023-10-05 21:42:01,837: INFO/ForkPoolWorker-14] Writing /tmp/tmp0otj9k4r/_remote_module_non_sriptable.py
# [2023-10-05 21:42:01,850: WARNING/ForkPoolWorker-14] Loading successful.
# [2023-10-05 21:42:01,863: INFO/ForkPoolWorker-14] Task tasks.predict_single[e53aa71f-c123-4c29-ad65-0cc3d4dc329e] succeeded in 0.03834275600092951s: {'label': 0, 'probs': [0.9983910918235779, 7.006843816270703e-08, 0.0002845293201971799, 4.518036291756289e-07, 3.631959998529055e-06, 0.00010714778181863949, 2.985075298056472e-05, 0.001065577263943851, 3.5141013086104067e-06, 0.00011397666821721941]}
# [2023-10-05 21:43:17,227: INFO/MainProcess] Task tasks.predict_single[70d49437-598d-4dfe-a898-1784b59ddb3e] received
# [2023-10-05 21:43:17,230: INFO/ForkPoolWorker-14] Task tasks.predict_single[70d49437-598d-4dfe-a898-1784b59ddb3e] succeeded in 0.0024355190034839325s: {'label': 0, 'probs': [0.9983910918235779, 7.006843816270703e-08, 0.0002845293201971799, 4.518036291756289e-07, 3.631959998529055e-06, 0.00010714778181863949, 2.985075298056472e-05, 0.001065577263943851, 3.5141013086104067e-06, 0.00011397666821721941]}

# bash try_api_get.sh "70d49437-598d-4dfe-a898-1784b59ddb3e" 
# {"task_id":"70d49437-598d-4dfe-a898-1784b59ddb3e",
#  "status":"complete",
#  "results":{"label":0,"probs":[0.9983910918235779,7.006843816270703e-08,0.0002845293201971799,4.518036291756289e-07,3.631959998529055e-06,0.00010714778181863949,2.985075298056472e-05,0.001065577263943851,3.5141013086104067e-06,0.00011397666821721941]}
