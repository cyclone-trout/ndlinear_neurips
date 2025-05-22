
from functools import partial

import torch


SOFTMAX_MAPPING = {
    "vanilla": torch.nn.functional.softmax,
}
