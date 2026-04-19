import cv2
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import numpy as np 
from math import log10, sqrt


def ssim_compare(img1, img2) :
    dim = (6022,5513)
    # print("Img1 Resolution:", img1.shape)
    # print("Img2 Resolution:", img2.shape)
    img1 = cv2.resize(img1, dim)
    img2 = cv2.resize(img2, dim)
    # print("Img1 Res :", img1.shape)
    # print("Img2 Res :", img2.shape)
    ssim_score, dif = ssim(img1, img2, full=True)
    return ssim_score

def PSNR (image_one, image_two):
    mse = np.mean((image_one - image_two) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

image = "outputs/I00451.jpg"
reconstructed = ["outputs/hed_result_thermal_three.png", "outputs/hed_result.png"]
ssim_val = ssim_compare(reconstructed[0], reconstructed[1])

print(ssim_val)

new_ssim_val = ssim_compare(image, reconstructed[0])
print(new_ssim_val)