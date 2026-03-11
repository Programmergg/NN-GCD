from torchvision import transforms
import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
from dataloader.transforms import *


def get_transform(transform_type='imagenet', image_size=32, args=None):

    if transform_type == 'imagenet':

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = args.interpolation
        crop_pct = args.crop_pct

        
        # train_transform = transforms.Compose([
        #     transforms.Resize(int(image_size / crop_pct), interpolation),
        #     transforms.RandomCrop(image_size),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     # transforms.RandomVerticalFlip(p=0.5),
        #     # transforms.RandomApply(
        #     #     [transforms.GaussianBlur(kernel_size=(5,5),sigma=(0.1,0.3)), ],
        #     #     p=0.5
        #     # ),
        #     # transforms.ColorJitter(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(
        #         mean=torch.tensor(mean),
        #         std=torch.tensor(std))
        # ])

        train_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomRotation(degrees=15),#cub imagenet
            #  transforms.RandomRotation(degrees=10),#cub
            # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.1),
            # transforms.RandomGrayscale(p=0.2),
            # GBlur(p=0.1),# 0.1
            # transforms.RandomApply([Solarization()], p=0.1),#p=0.2
            transforms.ToTensor(),  
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
        
        
        test_transform = transforms.Compose([
            transforms.Resize(int(image_size / crop_pct), interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
        # train_transforms_list = AMDIM_transforms(image_size)
        # test_transforms_list = test_transforms(image_size)
        # train_transform = transforms.Compose(
        #     [transforms.Resize(int(image_size / crop_pct), interpolation),
        #     transforms.RandomCrop(image_size)]+
        #     train_transforms_list + [transforms.Normalize(mean, std)])
        # test_transform = transforms.Compose(test_transforms_list + [transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    
    else:

        raise NotImplementedError

    return (train_transform, test_transform)

# _imagenet_pca = {
#     'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
#     'eigvec': torch.Tensor([
#         [-0.5675, 0.7192, 0.4009],
#         [-0.5808, -0.0045, -0.8140],
#         [-0.5836, -0.6948, 0.4203],
#     ])
# }
# class Solarization:
#     """Solarization as a callable object."""

#     def __call__(self, img: Image) -> Image:
#         """Applies solarization to an input image.

#         Args:
#             img (Image): an image in the PIL.Image format.

#         Returns:
#             Image: a solarized image.
#         """

#         return ImageOps.solarize(img)

class GBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img
    
# class GaussianBlur(object):
#     """blur a single image on CPU"""
#     def __init__(self, kernel_size):
#         radias = kernel_size // 2
#         kernel_size = radias * 2 + 1
#         self.blur_h = torch.nn.Conv2d(3, 3, kernel_size=(kernel_size, 1), stride=1, padding=0, bias=False, groups=3)
#         self.blur_v = torch.nn.Conv2d(3, 3, kernel_size=(1, kernel_size), stride=1, padding=0, bias=False, groups=3)
#         self.k = kernel_size
#         self.r = radias
#         self.blur = torch.nn.Sequential(
#             torch.nn.ReflectionPad2d(radias),
#             self.blur_h,
#             self.blur_v
#         )
#         self.pil_to_tensor = transforms.ToTensor()
#         self.tensor_to_pil = transforms.ToPILImage()

#     def __call__(self, img):
#         img = self.pil_to_tensor(img).unsqueeze(0)
#         sigma = np.random.uniform(0.1, 2.0)
#         x = np.arange(-self.r, self.r + 1)
#         x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
#         x = x / x.sum()
#         x = torch.from_numpy(x).view(1, -1).repeat(3, 1)
#         self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
#         self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))
#         with torch.no_grad():
#             img = self.blur(img)
#             img = img.squeeze()
#         img = self.tensor_to_pil(img)
#         return img

# class RandomTranslateWithReflect:
#     '''
#     Translate image randomly
#     Translate vertically and horizontally by n pixels where
#     n is integer drawn uniformly independently for each axis
#     from [-max_translation, max_translation].
#     Fill the uncovered blank area with reflect padding.
#     '''
#     def __init__(self, max_translation):
#         self.max_translation = max_translation

#     def __call__(self, old_image):
#         xtranslation, ytranslation = np.random.randint(-self.max_translation, self.max_translation + 1, size=2)
#         xpad, ypad = abs(xtranslation), abs(ytranslation)
#         xsize, ysize = old_image.size
#         flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
#         flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
#         flipped_both = old_image.transpose(Image.ROTATE_180)
#         new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))
#         new_image.paste(old_image, (xpad, ypad))
#         new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
#         new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))
#         new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
#         new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))
#         new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
#         new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
#         new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
#         new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))
#         new_image = new_image.crop((xpad - xtranslation, ypad - ytranslation,
#                                     xpad + xsize - xtranslation,
#                                     ypad + ysize - ytranslation))
#         return new_image
# def test_transforms(image_size):
#     transforms_list = [
#         transforms.Resize(256),
#         transforms.CenterCrop(image_size),
#         transforms.ToTensor(),
#     ]
#     return transforms_list

# def AMDIM_transforms(image_size):
#     flip_lr = transforms.RandomHorizontalFlip(p=0.5)
#     transforms_list = [
#         # transforms.RandomResizedCrop(image_size),
#         transforms.RandomApply([RandomTranslateWithReflect(8)], p=0.8),
#         # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
#         # transforms.RandomGrayscale(p=0.25),
#         transforms.ToTensor(),
#     ]
#     return transforms_list

# def SimCLR_transforms(image_size):
#     s = 1
#     color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
#     transforms_list = [
#         transforms.RandomResizedCrop(size=image_size),
#         transforms.RandomHorizontalFlip(),  # with 0.5 probability
#         transforms.RandomApply([color_jitter], p=0.8),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.ToTensor(),
#     ]
#     return transforms_list

# def AutoAug_transforms(image_size):
#     from dataloader.autoaug import RandAugment
#     transforms_list = [
#         # RandAugment(2, 12),
#         # ERandomCrop(image_size),
#         transforms.RandomHorizontalFlip(),
#         # transforms.ColorJitter(0.8, 0.8, 0.8),
#         transforms.ToTensor(),
#         Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
#     ]
#     return transforms_list

# def RandAug_transforms(image_size):
#     from dataloader.RandAugment import rand_augment_transform
#     rgb_mean = (0.485, 0.456, 0.406)
#     ra_params = dict(
#         translate_const=int(224 * 0.45),
#         img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]),
#     )
#     transforms_list = [
#         # transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
#         transforms.RandomHorizontalFlip(),
#         # transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
#         transforms.RandomApply([GaussianBlur(10)], p=0.5),
#         # rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
#         # transforms.RandomGrayscale(p=0.2),
#         transforms.ToTensor(),
#     ]
#     return transforms_list
