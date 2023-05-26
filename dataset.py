from __future__ import print_function, division
import torch
from skimage import io, transform, color
import numpy as np
import math
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class RescaleT(object):

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image, gt, mask = sample['image'], sample['gt'], sample['mask']

		h, w = image.shape[:2]

		if isinstance(self.output_size, int):
			if h > w:
				new_h, new_w = self.output_size*h/w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		img = transform.resize(image, (self.output_size, self.output_size), mode='constant')
		gt = transform.resize(gt, (self.output_size, self.output_size), mode='constant', order=0, preserve_range=True)
		mask = transform.resize(mask, (self.output_size, self.output_size), mode='constant', order=0, preserve_range=True)

		return {'image': img, 'gt': gt, 'mask': mask}


class RandomCrop(object):

	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		if isinstance(output_size, int):
			self.output_size = (output_size, output_size)
		else:
			assert len(output_size) == 2
			self.output_size = output_size

	def __call__(self, sample):
		image, gt, mask = sample['image'], sample['gt'], sample['mask']

		h, w = image.shape[:2]
		new_h, new_w = self.output_size

		# top = np.random.randint(0, new_h - h)
		# left = np.random.randint(0, new_w - w)

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)

		image = image[top: top + new_h, left: left + new_w]
		gt = gt[top: top + new_h, left: left + new_w]
		mask = mask[top: top + new_h, left: left + new_w]

		return {'image': image, 'gt': gt, 'mask':mask}


class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self, flag=0):
		self.flag = flag

	def __call__(self, sample):

		image, gt, mask = sample['image'], sample['gt'], sample['mask']

		tmpgt = np.zeros(gt.shape)

		if np.max(gt) < 1e-6:
			gt = gt
		else:
			gt = gt/np.max(gt)

		tmpmask = np.zeros(mask.shape)

		if np.max(mask) < 1e-6:
			mask = mask
		else:
			mask = mask/np.max(mask)

		# change the color space
		if self.flag == 2:  # with rgb and Lab colors
			tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
			tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
			if image.shape[2] == 1:
				tmpImgt[:, :, 0] = image[:, :, 0]
				tmpImgt[:, :, 1] = image[:, :, 0]
				tmpImgt[:, :, 2] = image[:, :, 0]
			else:
				tmpImgt = image
			tmpImgtl = color.rgb2lab(tmpImgt)

			tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
			tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
			tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
			tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
			tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
			tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
			tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
			tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
			tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

		elif self.flag == 1:  #with Lab color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))

			if image.shape[2] == 1:
				tmpImg[:,:,0] = image[:,:,0]
				tmpImg[:,:,1] = image[:,:,0]
				tmpImg[:,:,2] = image[:,:,0]
			else:
				tmpImg = image

			tmpImg = color.rgb2lab(tmpImg)

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

		else: # with rgb color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))
			image = image/np.max(image)
			if image.shape[2] == 1:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
			else:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
				tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpgt[:,:,0] = gt[:,:,0]
		tmpmask[:, :, 0] = mask[:, :, 0]

		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpgt = gt.transpose((2, 0, 1))
		tmpmask = mask.transpose((2, 0, 1))

		return {'image': torch.from_numpy(tmpImg), 'gt': torch.from_numpy(tmpgt), 'mask': torch.from_numpy(tmpmask)}


class SalObjDataset(Dataset):
	def __init__(self, img_name_list, gt_name_list, mask_name_list, transform=None):
		self.image_name_list = img_name_list
		self.gt_name_list = gt_name_list
		self.mask_name_list = mask_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self, idx):

		image = io.imread(self.image_name_list[idx])

		if(0==len(self.gt_name_list)):
			gt_3 = np.zeros(image.shape)
		else:
			gt_3 = io.imread(self.gt_name_list[idx])

		gt = np.zeros(gt_3.shape[0:2])
		if(3==len(gt_3.shape)):
			gt = gt_3[:,:,0]
		elif(2==len(gt_3.shape)):
			gt = gt_3

		if(3==len(image.shape) and 2==len(gt.shape)):
			gt = gt[:,:,np.newaxis]
		elif(2==len(image.shape) and 2==len(gt.shape)):

			gt = gt[:,:,np.newaxis]

		if(0==len(self.mask_name_list)):
			mask_3 = np.zeros(image.shape)
		else:
			mask_3 = io.imread(self.mask_name_list[idx])

		mask = np.zeros(mask_3.shape[0:2])
		if(3==len(mask_3.shape)):
			mask = mask_3[:,:,0]
		elif(2==len(mask_3.shape)):
			mask = mask_3

		if(3==len(image.shape) and 2==len(mask.shape)):
			mask = mask[:,:,np.newaxis]
		elif(2==len(image.shape) and 2==len(mask.shape)):
			image = image[:,:,np.newaxis]
			mask = mask[:,:,np.newaxis]

		sample = {'image':image, 'gt':gt, 'mask':mask}

		if self.transform:
			sample = self.transform(sample)

		return sample
