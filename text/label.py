import pixelhouse as ph
import numpy as np
import cv2


def displace(text, x=-1.25, y=2, dx=10):
    
    dye = ph.Canvas(1920, 1080)
    
    dye += ph.text(
        text, x=x,y=y,font_size=0.35,font="VCR_OSD_MONO_1.001.ttf")

    if not dx:
        return dye

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

    return dyex



text = "Hack&&Tell NewNewYork Demo"

dye = ph.Canvas(1920, 1080)
for y,dx in zip(range(2, -3, -1), [10,5,3,2,0]):
    dye += displace(text, y=y*0.6, dx=dx)

blur_amount = 0.04
dye += ph.filters.gaussian_blur(blur_amount, blur_amount)

dye.save("title.png")
dye.resize(0.5).show()
