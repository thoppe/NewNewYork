import os, glob, random
import numpy as np
from tqdm import tqdm
import pixelhouse.pixelhouse as ph

import cv2
rgb_pixel_offset = dx = 2
blur_amount = 0.1


save_dest = "data/overlay"
os.system(f"mkdir -p {save_dest}")

dmap_dest = 'data/dmaps/'
F_JPG = sorted(glob.glob("data/frames/*"))
random.shuffle(F_JPG)

for f0 in tqdm(F_JPG):
    
    f2 = os.path.join("data/frame_dye/", os.path.basename(f0))
    fd = os.path.join(dmap_dest, os.path.basename(f0)+'.npy')
    f_save = os.path.join(save_dest, os.path.basename(f0))
    
    if not os.path.exists(f2):
        continue

    if not os.path.exists(fd):
        continue

    if os.path.exists(f_save):
        continue

    print(f_save)

    dm = np.load(fd)

    # Clip and scale
    min_dm, max_dm = 1, 3
    dm = np.clip(dm, min_dm, max_dm)
    dm = (dm-min_dm)/(max_dm-min_dm)
    alpha = np.clip(255*(1-dm), 0, 255).astype(np.uint8)

    dye = ph.load(f2)
    dye += ph.filters.gaussian_blur(blur_amount, blur_amount)

    with dye.layer() as layer:
        org = ph.load(f0)
        org.alpha = alpha
        layer += org

    #canvas.resize(0.5).show()
    #ph.vstack([dye, ph.load(f0)]).resize(0.4).show(1)
    

    dyeR = dye.copy(); dyeR.img[:,:,1] = 0; dyeR.img[:,:,2] = 0
    dyeG = dye.copy(); dyeG.img[:,:,0] = 0; dyeG.img[:,:,2] = 0
    dyeB = dye.copy(); dyeB.img[:,:,0] = 0; dyeB.img[:,:,1] = 0   

    M = np.float32([[1,0,dx],[0,1,dx]])
    dyeR.img = cv2.warpAffine(dyeR.img,M,(dye.shape[1], dye.shape[0]))

    M = np.float32([[1,0,dx],[0,1,-dx]])
    dyeG.img = cv2.warpAffine(dyeG.img,M,(dye.shape[1], dye.shape[0]))

    M = np.float32([[1,0,-dx],[0,1,0]])
    dyeB.img = cv2.warpAffine(dyeB.img,M,(dye.shape[1], dye.shape[0]))

    dyex = dye.blank()
    dyex.img += dyeR.img
    dyex.img += dyeG.img
    dyex.img += dyeB.img

    # Fix the corners
    dyex.img[0:dx, :, :] += dyeR.img[dx:2*dx, :, :]
    dyex.img[:, 0:dx, :] += dyeR.img[:, dx:2*dx, :]

    dyex.img[:, 0:dx, :] += dyeG.img[:, dx:2*dx, :]
    dyex.img[-dx:, :, :] += dyeG.img[-2*dx:-dx, :, :]

    dyex.img[:, -dx:, :] += dyeB.img[:, -2*dx:-dx, :]
    
    #ph.vstack([dyex, ph.load(f0)]).resize(0.4).show(1)
    dyex.save(f_save)

    
