import cv2
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim


def ssim_compare(img1_path, img2_path) :
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)
    dim = (6022,5513)
    # print("Img1 Resolution:", img1.shape)
    # print("Img2 Resolution:", img2.shape)
    img1 = cv2.resize(img1, dim)
    img2 = cv2.resize(img2, dim)
    # print("Img1 Res :", img1.shape)
    # print("Img2 Res :", img2.shape)
    ssim_score, dif = ssim(img1, img2, full=True)
    return ssim_score

ssim_val = 0
for i in range(10):
    ssim_val += ssim_compare(f'outputs/edges/edges_visible_{i}.png', f'outputs/edges/edges_thermal_{i}.png')

print(ssim_val / 10)