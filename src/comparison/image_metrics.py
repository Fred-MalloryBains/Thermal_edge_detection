import cv2
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import numpy as np 
from math import log10, sqrt
import matplotlib.pyplot as plt
import matplotlib as mpl


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

def save_best_stems_grid(best_stems, base_path, save_path="outputs/edges_hed_comp/best_grid.png"):
    
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
    })

    fig, axes = plt.subplots(3, 4, figsize=(10, 7.5))

    # --- Minimal whitespace ---
    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.95,
        bottom=0.05,
        wspace=0.01,
        hspace=0.05
    )

    for row, stem in enumerate(best_stems):
        try:
            # --- Load images ---
            gt = cv2.imread(
                f"outputs/edges_hed_comp/edges_hed{stem}_gt.png",
                cv2.IMREAD_GRAYSCALE
            )
            thermal = cv2.imread(
                f"outputs/edges_hed_comp/edges_hed{stem}_thermal.png",
                cv2.IMREAD_GRAYSCALE
            )
            comp = cv2.imread(
                f"outputs/edges_hed_custom/edges_hed{stem}.png",
                cv2.IMREAD_GRAYSCALE
            )
            original = cv2.imread(
                f"{base_path}/{stem}.jpg"
            )
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

            # --- Normalize edge maps consistently ---
            stack = np.stack([gt, thermal, comp], axis=0).astype(np.float32)
            vmin, vmax = stack.min(), stack.max()

            def norm(img):
                return (img - vmin) / (vmax - vmin + 1e-8)

            gt_n = norm(gt)
            thermal_n = norm(thermal)
            comp_n = norm(comp)

            # --- Metrics ---
            ssim_base = ssim_compare(gt, thermal)
            ssim_comp = ssim_compare(gt, comp)

            images = [original, gt_n, thermal_n, comp_n]

            for col in range(4):
                ax = axes[row, col]

                if col == 0:
                    ax.imshow(images[col])  # RGB image
                else:
                    ax.imshow(images[col], cmap="gray", vmin=0, vmax=1)

                ax.axis("off")

                # Column titles (top row only)
                if row == 0:
                    titles = ["Original", "Ground Truth", "Thermal HED", "Custom HED"]
                    ax.set_title(titles[col])

                # SSIM under relevant columns
                if col == 2:
                    ax.text(
                        0.5, -0.10,
                        f"{ssim_base:.3f}",
                        transform=ax.transAxes,
                        ha="center",
                        fontsize=8
                    )
                elif col == 3:
                    ax.text(
                        0.5, -0.10,
                        f"{ssim_comp:.3f}",
                        transform=ax.transAxes,
                        ha="center",
                        fontsize=8
                    )

        except Exception as e:
            print(f"Skipping {stem}: {e}")

    # Save high-quality figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


stems = ["I01029", "I00323", "I00337", "I00445", "I00479", "I00486", "I00492", "I01001"]
best_stems = ["I01029", "I01001", "I00323"]
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

save_best_stems_grid(best_stems, base_path)
    
    