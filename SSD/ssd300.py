import torch
import torch.nn as nn
from base_model import VGG16

class VGG16Extractor300(nn.Module):
    def __init__(self):
        super(VGG16Extractor300, self).__init__()
        # input: 3*300*300 --(VGG16)--> 512*38*38
