import pixelhouse.pixelhouse as ph
import glob, os
import pyflow
from tqdm import tqdm
from PIL import Image
import numpy as np
import time

F_FRAMES = sorted(glob.glob("data/frames/*"))[3072:]

save_dest = "data/flows"
os.system(f'mkdir -p {save_dest}')

# Flow Options:
alpha = 0.05 ## SCALE PARAM
ratio = 0.75
minWidth = 200
nOuterFPIterations = 1
nInnerFPIterations = 1
nSORIterations = 1
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

frac = 0.25
frac = 0.5
for f0, f0b in zip(tqdm(F_FRAMES), F_FRAMES[1:]):

    #print(f0)
    
    f1 = os.path.join(save_dest, os.path.basename(f0)) + '.npy'

    #im1 = np.array(Image.open(f0))
    #im2 = np.array(Image.open(f0b))

    im1 = ph.load(f0)
    im2 = ph.load(f0b)
    img = im1.copy()
    img.img = np.abs(im1.img-im2.img)

    img.show()
    continue
    exit()

    im1 = ph.load(f0).resize(frac).rgb
    im2 = ph.load(f0b).resize(frac).rgb
    
    im1 = im1.astype(float) / 255.
    im2 = im2.astype(float) / 255.

    s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations,
        nInnerFPIterations,
        nSORIterations, colType)
    e = time.time()
    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    print(flow, flow.shape)
    np.save(f1, flow)
    #np.save('examples/outFlow.npy', flow)

    import cv2
    hsv = np.zeros(im1.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    print(mag, mag.shape)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    #mag = np.clip(100*mag, 0, 255)
    #mag = mag.astype(np.uint8)
    print(mag.shape)
    print(mag)
        
    canvas = ph.Canvas()
    canvas.img = np.dstack([mag,mag,mag,])
    canvas.resize(1.0).show()
    #exit()
