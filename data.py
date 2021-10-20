import numpy as np
import nibabel as nib
from skimage import exposure
import matplotlib.pyplot as plt # plt 用于显示图片
from data_gen import *
def bn(X):
	x = np.reshape(X,(224*224*128,1))
	max = np.max(x)
	min = np.min(x)
	x = (x-min)/(max-min)
	x = np.reshape(x,(224,224,128))
	return x
def random(x_train,x_mask):
	permutation = np.random.permutation(x_mask.shape[0])
	x_train = x_train[permutation, :, :] #训练数据
	x_mask = x_mask[permutation] #训练标签
	return x_train,x_mask
def add_rice_noise(img, snr=1, mu=0.0, sigma=1):
	level = snr * np.max(img) / 100
	size = img.shape
	x = level * np.random.normal(mu, sigma, size=size) + img
	y = level * np.random.normal(mu, sigma, size=size)
	return np.sqrt(x**2 + y**2)

def randomHorizontalFlip(image, mask, u=0.5):
	if np.random.random() < u:
		image = np.flip(image, axis=(1))
		mask = np.flip(mask, axis=(1))
	return image, mask

def randomVerticalFlip(image, mask, u=0.5):
	if np.random.random() < u:
		image = np.flip(image, axis=(2))
		mask = np.flip(mask, axis=(2))
	return image, mask

def randomRotation(image, mask, u=0.5):
	n_rand = np.random.random()
	if n_rand < u:
		if n_rand <= u/3:
			k=1
		elif n_rand > u - u/3:
			k=3
		else:
			k=2
		image = np.rot90(image, k=k, axes=(1,2))
		mask = np.rot90(mask, k=k, axes=(1,2))
	return image, mask
def argument(image,mask,axis):
	image = np.flip(image,axis=axis)
	mask = np.flip(mask,axis=axis)
	return image,mask
def generator_noise_data(image_file,st,image_mask_file,sl):
	vol_dir = image_file + st
	image1 = nib.load(vol_dir)
	image = image1.get_data()
	affine0 = image1.affine.copy()
	image = np.asarray(image, dtype=np.float32)
	#image = image[16:240,16:240,:]
	mask_dir = image_mask_file + sl
	mask1 = nib.load(mask_dir)
	mask = mask1.get_data()
	affine0 = mask1.affine.copy()
	mask = np.asarray(mask, dtype=np.float32)
	#mask = mask[16:240,16:240,:]
	#mask = bn(mask)
	#noise_image = add_rice_noise(image,snr = noise_level)
	return image,mask,affine0
def generator_test_noise_data(image_file,st,image_mask_file,sl,train_skull_file,ku):
	vol_dir = image_file + st
	image1 = nib.load(vol_dir)
	image = image1.get_data()
	affine0 = image1.affine.copy()
	image = np.asarray(image, dtype=np.float32)
	#image = image[16:240,16:240,:]
	mask_dir = image_mask_file + sl
	mask1 = nib.load(mask_dir)
	mask = mask1.get_data()
	affine0 = mask1.affine.copy()
	mask = np.asarray(mask, dtype=np.float32)
	skull_dir = train_skull_file + ku
	skull1 = nib.load(skull_dir)
	skull = skull1.get_data()
	skull = np.asarray(skull, dtype=np.float32)
	return image,mask,skull,affine0
def get_test_data(base_dir,image_file,label_file,txt_file):
	train_file = open(base_dir+txt_file)  # 训练数据的名字放到txt文件里
	train_strings = train_file.readlines()
	image = []
	test_label = []
	affine = []
	for i in range(0,len(train_strings)):
		st = train_strings[i].strip()  #文件名
		img,mask1,affine1= generator_noise_data(image_file,st,label_file,st)
		image.append(img)
		test_label.append(mask1)
		affine.append(affine1)
	image = np.asarray(image, dtype=np.float32)
	test_label = np.asarray(test_label, dtype=np.float32)
	return image,test_label,affine
def get_test_BM3D_data(base_dir,image_file,label_file,txt_file,label_txt_file):
	train_file = open(base_dir+txt_file)  # 训练数据的名字放到txt文件里
	train_strings = train_file.readlines()
	train_label_file = open(base_dir+label_txt_file)  # 训练数据的名字放到txt文件里
	train_label_strings = train_label_file.readlines()
	image = []
	test_label = []
	affine = []
	for i in range(0,len(train_strings)):
		st = train_strings[i].strip()  #文件名
		sl = train_label_strings[i].strip()
		img,mask1,affine1= generator_noise_data(image_file,st,label_file,sl)
		image.append(img)
		test_label.append(mask1)
		affine.append(affine1)
	image = np.asarray(image, dtype=np.float32)
	test_label = np.asarray(test_label, dtype=np.float32)
	return image,test_label,affine
def random_data(x_train,x_mask):
	permutation = np.random.permutation(x_mask.shape[0])
	x_train = x_train[permutation, :, :] #训练数据
	x_mask = x_mask[permutation, :, :]
	return x_train,x_mask
def generator_patch(x,x2):
	train_patch = vols_generator_patch(vol_name=x, num_data=1, patch_size=[64,64,64],
									   stride_patch=[32,32,64], out=1, num_images=49)
	mask_patch = vols_generator_patch(vol_name=x2, num_data=1, patch_size=[64,64,64],
									  stride_patch=[32,32,64], out=1, num_images=49)
	train_patch, mask_patch = random_data(train_patch,mask_patch)
	return train_patch,mask_patch

def BatchGenerator(train_file,train_mask_file,txt_file,txt_mask_file,batch_size=1,batch_patch_size=1,augment=True):
	while True:
		image_file = open(txt_file)  # 训练数据的名字放到txt文件里
		image_strings = image_file.readlines()
		mask_file = open(txt_mask_file)
		mask_strings = mask_file.readlines()
		for start in range(0, len(image_strings), batch_size):
			image_batch = []
			mask_batch = []
			end = min(start + batch_size,len(image_strings))
			for id in range(start, end):
				st = image_strings[id].strip()  # 文件名
				sl = mask_strings[id].strip()
				image,mask,affine = generator_noise_data(train_file,st,train_mask_file,sl)
				#nib.save(nib.Nifti1Image(image,affine),'1.nii.gz')
				#nib.save(nib.Nifti1Image(mask,affine),'2.nii.gz')
				if augment:
					image, mask = argument(image, mask, axis=0)
					#image, mask = randomRotation(image, mask)
					image, mask = argument(image, mask, axis=2)
					image, mask = argument(image, mask, axis=1)
				image_batch.append(image)
				mask_batch.append(mask)
			x_image_patch, x_mask_patch = generator_patch(image_batch,mask_batch)
			for start in range(0, len(x_image_patch), batch_patch_size):
				x_patch = []
				y_patch = []
				end = min(start + batch_patch_size, len(x_image_patch))
				for id in range(start, end):
					x_patch1 = x_image_patch[id]
					y_patch1 = x_mask_patch[id]
					x_patch.append(x_patch1)
					y_patch.append(y_patch1)
				x_patch = np.array(x_patch)
				y_patch = np.array(y_patch)
				x_patch = x_patch[..., np.newaxis]
				y_patch = y_patch[..., np.newaxis]
				yield x_patch,y_patch
def BatchGenerator2D(train_file,train_mask_file,txt_file,txt_mask_file,batch_size=1,batch_patch_size=1,augment=True):
	while True:
		image_file = open(txt_file)  # 训练数据的名字放到txt文件里
		image_strings = image_file.readlines()
		mask_file = open(txt_mask_file)
		mask_strings = mask_file.readlines()
		for start in range(0, len(image_strings), batch_size):
			image_batch = []
			mask_batch = []
			end = min(start + batch_size,len(image_strings))
			for id in range(start, end):
				st = image_strings[id].strip()  # 文件名
				sl = mask_strings[id].strip()
				image,mask,affine = generator_noise_data(train_file,st,train_mask_file,sl)
				image = np.transpose(image,(2,1,0))
				mask = np.transpose(mask,(2,1,0))
				#nib.save(nib.Nifti1Image(image,affine),'1.nii.gz')
				#nib.save(nib.Nifti1Image(mask,affine),'2.nii.gz')
				if augment:
					image, mask = argument(image, mask, axis=0)
					#image, mask = randomRotation(image, mask)
					image, mask = argument(image, mask, axis=2)
					image, mask = argument(image, mask, axis=1)
				image_batch.append(image)
				mask_batch.append(mask)
			x_image_patch = np.array(image_batch)
			x_mask_patch = np.array(mask_batch)
			x_image_patch = np.squeeze(x_image_patch,axis=0)
			x_mask_patch = np.squeeze(x_mask_patch,axis=0)
			x_image_patch,x_mask_patch = random(x_image_patch,x_mask_patch )
			for start in range(0, len(x_image_patch), batch_patch_size):
				x_patch = []
				y_patch = []
				end = min(start + batch_patch_size, len(x_image_patch))
				for id in range(start, end):
					x_patch1 = x_image_patch[id]
					y_patch1 = x_mask_patch[id]
					x_patch.append(x_patch1)
					y_patch.append(y_patch1)
				x_patch = np.array(x_patch)
				y_patch = np.array(y_patch)
				x_patch = x_patch[..., np.newaxis]
				y_patch = y_patch[..., np.newaxis]
				yield x_patch,y_patch