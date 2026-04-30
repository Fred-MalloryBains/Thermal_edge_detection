import cv2
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import numpy as np 
from math import log10, sqrt
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


def ssim_compare(img1, img2) :
    dim = (6022,5513)
    # print("Img1 Resolution:", img1.shape)
    # print("Img2 Resolution:", img2.shape)
    img1 = cv2.resize(img1, dim)
    img2 = cv2.resize(img2, dim)
    # print("Img1 Res :", img1.shape)
    # print("Img2 Res :", img2.shape)
    ssim_score, dif = ssim(img1, img2, full=True, channel_axis=2)
    return ssim_score

def PSNR (image_one, image_two):
    if image_one.shape != image_two.shape:
        image_two = cv2.resize(image_two, (image_one.shape[1], image_one.shape[0]))

    mse = np.mean((image_one - image_two) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_best_stems_grid(best_stems, save_path="outputs/final_comparison/best_grid.png"):

    mpl.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 9})

    print(f"Number of stems: {len(best_stems)}")  # ← check this, bet it's 4

    n = len(best_stems)
    fig, axes = plt.subplots(n, 6, figsize=(12, (12 / 6) * n))
    if n == 1:
        axes = axes[None, :]

    plt.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.05, wspace=0.01, hspace=0.05)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    titles = ["Ground Truth", "Thermal input", "Baseline Edge", "Our HED", "Reconstruction w/o token", "Reconstruction w/ token"]
    
    base = "outputs/final_comparison"

    for row, stem in enumerate(best_stems):
        print(f"[{row}] Processing stem: {stem}")

        gt    = cv2.imread(f"{base}/gt/{stem}.png")
        thermal = cv2.imread(f"{base}/thermal/{stem}.png")
        edge_base = cv2.imread(f"{base}/base_edges/{stem}_base.png")
        edge_hed = cv2.imread(f"{base}/base_edges/{stem}_model.png")
        token = cv2.imread(f"{base}/recon/{stem}_tokens.png")
        recon  = cv2.imread(f"{base}/recon/{stem}.png")

        
        for col, img in enumerate([gt, thermal, edge_base, edge_hed, recon, token]):
            cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA, dst=img)
            ax = axes[row, col]
            ax.axis("off")
            if row == 0:
                ax.set_title(titles[col])
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


stems = ["I01029", "I00323", "I00337", "I00445", "I00479", "I00486", "I00492", "I01001"]
best_stems = ["I01029", "I01001", "I00323"]
inversion_stems = ["I015", "I0047", "I00337"]
recon_stems = ["I01035", "I02509", "I00000"]
base_path = "/Volumes/Samsung_1TB/thermal_images/archive/set00/V007/visible/"

"""
for stem in best_stems:
    try:
        ground_truth = cv2.imread(f"outputs/edges_hed_comp/edges_hed{stem}_gt.png", cv2.IMREAD_GRAYSCALE)
        thermal_edge = cv2.imread(f"outputs/edges_hed_comp/edges_hed{stem}_thermal.png", cv2.IMREAD_GRAYSCALE)
        comp = cv2.imread(f"outputs/edges_hed_custom/edges_hed{stem}.png", cv2.IMREAD_GRAYSCALE)
        
        ssim_val = ssim_compare(ground_truth, thermal_edge)
        psnr = PSNR(ground_truth, thermal_edge)
        

        new_ssim_val = ssim_compare(ground_truth, comp)
        psnr_comp = PSNR(ground_truth, comp)

        print(f"Image: {stem} | SSIM truth {ssim_val:.4f} |  SSIM comp {new_ssim_val:.4f}")

    except Exception as e:
        pass
"""
base = "outputs/final_comparison"
for stem in recon_stems:
    try:
        gt    = cv2.imread(f"{base}/gt/{stem}.png")
        thermal = cv2.imread(f"{base}/thermal/{stem}.png")
        edge_base = cv2.imread(f"{base}/base_edges/{stem}_base.png")
        edge_hed = cv2.imread(f"{base}/base_edges/{stem}_model.png")
        token = cv2.imread(f"{base}/recon/{stem}_tokens.png")
        recon  = cv2.imread(f"{base}/recon/{stem}.png")

        ssim_recon = ssim_compare(gt, recon)
        psnr_recon = PSNR(gt, recon)

        ssim_comp = ssim_compare(gt, token)
        psnr_comp = PSNR(gt, token)

        print(f"Stem: {stem} | SSIM Recon {ssim_recon:.4f} | PSNR Recon {psnr_recon:.2f} | SSIM Token {ssim_comp:.4f} | PSNR Token {psnr_comp:.2f}")

    except Exception as e:
        print(f"Error processing stem {stem}: {e}")

#save_best_stems_grid(recon_stems)
    
    
