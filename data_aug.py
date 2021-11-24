import cv2
import random
import PIL.Image
import numpy as np
import torchvision.transforms as transforms

class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = self.size
        w, h = im.shape[1], im.shape[0]

        if (W, H) == (w, h): return dict(im=im, lb=lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            lb = cv2.resize(lb, (w, h), interpolation=cv2.INTER_NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H

        return dict(
                im = im[crop[1]:crop[3], crop[0]:crop[2]],
                lb = lb[crop[1]:crop[3], crop[0]:crop[2]]
                    )

class HorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']

            flip_lb = np.array(lb)
            flip_lb[lb == 2] = 3
            flip_lb[lb == 3] = 2
            flip_lb[lb == 4] = 5
            flip_lb[lb == 5] = 4
            flip_lb[lb == 7] = 8
            flip_lb[lb == 8] = 7

            return dict(im = cv2.flip(im, 1),
                        lb = cv2.flip(flip_lb, 1),
                    )

class RandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = im.shape[1], im.shape[0]
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        return dict(im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR),
                    lb = cv2.resize(lb, (w, h), interpolation=cv2.INTER_NEAREST),
                )

class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        self.jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)

    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        
        im = PIL.Image.fromarray(im)
        im = self.jitter(im)
        im = np.array(im)

        return dict(im = im,
                    lb = lb,
                )