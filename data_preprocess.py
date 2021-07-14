import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

face_data = '/home/seongjae/Downloads/CelebAMask-HQ/CelebA-HQ-img'
face_sep_mask = '/home/seongjae/Downloads/CelebAMask-HQ/CelebAMask-HQ-mask-anno'
mask_path = '/home/seongjae/Downloads/CelebAMask-HQ/mask'
# mask_path = '.'
counter = 0
total = 0

atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

num_partiton = 15
data_per_partition = 2000

for i in tqdm(range(num_partiton)):
    for j in tqdm(range(i*data_per_partition, (i+1)*data_per_partition)):
        mask = np.zeros((512, 512))

        for l, att in enumerate(atts, 1):
            total += 1
            file_name = f'{j:05}_{att}.png'

            path = os.path.join(face_sep_mask, str(i), file_name)

            if os.path.exists(path):
                counter += 1
                sep_mask = np.array(Image.open(path).convert('P'))
                mask[sep_mask == 225] = l

        cv2.imwrite(f'{mask_path}/{j}.png', mask)

print(counter, total)
