import os
import torch
import torchvision
import segmentation_models_pytorch as smp

from torchvision import transforms

from tqdm import tqdm
from glob import glob

# Load images
img_paths = sorted(glob('samples/*'))
print(img_paths)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
print(transform)

# Define model
import sys
sys.path.insert(0,'..')
from trainer import SegmentModel
model = SegmentModel.load_from_checkpoint('../checkpoints/epoch=84-step=148749.ckpt')
print(model)

# Test inference

# Lib color changer

# Loop over test imgs

