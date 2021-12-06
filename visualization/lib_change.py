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
model.cuda()
model.eval()
print(model)

# Lib color changer
# atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
#         10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
def lib_color_change(img, segment):
    print(img.shape)
    print(segment.shape)
    
    mask = (segment == 12) | (segment == 13)
    img[..., 0][mask] += 50

    return img

# Test inference
import cv2
import numpy as np
import face_recognition
import sys
sys.path.append('../')
from utils import labelVisualize

for img_path in tqdm(img_paths):
    img = face_recognition.load_image_file(img_path)
    face_locations = face_recognition.face_locations(img)
    face_location = face_locations[0]
    top, right, bottom, left = face_location
    img = img[top:bottom, left:right]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(type(img))
    print(img.shape)
    print(face_locations)

    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # img_t = img[None, :, :, :]
    # print(img_t.shape)
    img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    img_t = transform(img).unsqueeze(0)
    print(img_t)
    print(img_t.shape)
    out_t = model(img_t.cuda())
    print(out_t.shape)
    out = out_t[0].cpu().numpy()
    print(out.shape)

    out_v = labelVisualize(out, num_class=18)
    out_v = np.uint8(out_v * 255.0)

    print(img.shape)
    print(out_v.shape)

    img_result = lib_color_change(img.copy(), out)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', np.hstack([img, out_v, img_result]))
    cv2.waitKey(0)


    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img.shape)

    # img_t = transform(img)
    # print(img_t.shape)

    break
