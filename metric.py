from keras import backend as K
import math
import numpy as np
import tensorflow as tf
import scipy
import scipy.ndimage
from skimage.measure import compare_ssim
from skimage import  exposure
from scipy.ndimage import imread
from numpy.ma.core import exp
from scipy.constants.constants import pi
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred -y_true))))

def ssim(ts,ps):
    ssim_v = tf.reduce_sum(tf.image.ssim(ts, ps, 1.0))
    return ssim_v
def psnr(img1, img2):
   mse = np.mean((img1/1.0 - img2/1.0) ** 2 )
   if mse < 1.0e-10:
      return 100
   psnr = 10 * math.log10(1**2/mse)
   return mse,psnr


def compute_ssim(img_mat_1, img_mat_2):
    # Variables for Gaussian kernel definition
    gaussian_kernel_sigma = 1.5
    gaussian_kernel_width = 11
    gaussian_kernel = np.zeros((gaussian_kernel_width, gaussian_kernel_width))

    # Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i, j] = \
                (1 / (2 * pi * (gaussian_kernel_sigma ** 2))) * \
                exp(-(((i - 5) ** 2) + ((j - 5) ** 2)) / (2 * (gaussian_kernel_sigma ** 2)))

    # Convert image matrices to double precision (like in the Matlab version)
    #img_mat_1 = img_mat_1.astype(np.float)
    #img_mat_2 = img_mat_2.astype(np.float)

    # Squares of input matrices
    img_mat_1_sq = img_mat_1 ** 2
    img_mat_2_sq = img_mat_2 ** 2
    img_mat_12 = img_mat_1 * img_mat_2

    # Means obtained by Gaussian filtering of inputs
    img_mat_mu_1 = scipy.ndimage.filters.convolve(img_mat_1, gaussian_kernel)
    img_mat_mu_2 = scipy.ndimage.filters.convolve(img_mat_2, gaussian_kernel)

    # Squares of means
    img_mat_mu_1_sq = img_mat_mu_1 ** 2
    img_mat_mu_2_sq = img_mat_mu_2 ** 2
    img_mat_mu_12 = img_mat_mu_1 * img_mat_mu_2

    # Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq = scipy.ndimage.filters.convolve(img_mat_1_sq, gaussian_kernel)
    img_mat_sigma_2_sq = scipy.ndimage.filters.convolve(img_mat_2_sq, gaussian_kernel)

    # Covariance
    img_mat_sigma_12 = scipy.ndimage.filters.convolve(img_mat_12, gaussian_kernel)

    # Centered squares of variances
    img_mat_sigma_1_sq = img_mat_sigma_1_sq - img_mat_mu_1_sq
    img_mat_sigma_2_sq = img_mat_sigma_2_sq - img_mat_mu_2_sq
    img_mat_sigma_12 = img_mat_sigma_12 - img_mat_mu_12

    # c1/c2 constants
    # First use: manual fitting
    c_1 = 6.5025
    c_2 = 58.5225

    # Second use: change k1,k2 & c1,c2 depend on L (width of color map)
    l = 255
    k_1 = 0.01
    c_1 = (k_1 * l) ** 2
    k_2 = 0.03
    c_2 = (k_2 * l) ** 2

    # Numerator of SSIM
    num_ssim = (2 * img_mat_mu_12 + c_1) * (2 * img_mat_sigma_12 + c_2)
    # Denominator of SSIM
    den_ssim = (img_mat_mu_1_sq + img_mat_mu_2_sq + c_1) * \
               (img_mat_sigma_1_sq + img_mat_sigma_2_sq + c_2)
    # SSIM
    ssim_map = num_ssim / den_ssim
    index = np.average(ssim_map)

    return index
def entropy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    img = img * 255
    img = np.asarray(img, dtype=np.uint8)
    img = exposure.adjust_gamma(img, 0.3)  # 调暗
    out = 0
    count = np.shape(img)[0]*np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i]!=0:
            out-=p[i]*math.log(p[i]/count)/count
    return out

