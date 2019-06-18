import pixelhouse.pixelhouse as ph
import glob, os
from tqdm import tqdm
import random

F_FRAMES = sorted(glob.glob("data/frames/*"))
random.shuffle(F_FRAMES)

save_dest = "data/frame_dye"
os.system(f'mkdir -p {save_dest}')

for f0 in tqdm(F_FRAMES):
    #f0 = "data/frames/000715.jpg"
    f1 = os.path.join(save_dest, os.path.basename(f0))
    if os.path.exists(f1): continue
    
    canvas = ph.load(f0)#.resize(0.5)
    canvas += ph.filters.instafilter("NeveraSleep")
    canvas.save(f1)

    
    print(f1)

