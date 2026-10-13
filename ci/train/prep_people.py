# Collect people photos (faces + full-body pedestrians) from public datasets into /tmp/people/
import os, io, zipfile, urllib.request, numpy as np
from PIL import Image
out='/tmp/people'; os.makedirs(out, exist_ok=True); n=0
try:
    from sklearn.datasets import fetch_lfw_people
    d=fetch_lfw_people(color=True, slice_=None, resize=1.0, min_faces_per_person=1, download_if_missing=True)
    imgs=d.images; idx=np.random.RandomState(0).permutation(len(imgs))[:2500]
    for k,i in enumerate(idx):
        a=imgs[i]; a=(a*255 if a.max()<=1.5 else a).astype(np.uint8); Image.fromarray(a).save(f'{out}/lfw_{k}.jpg',quality=95); n+=1
    print('lfw ok',len(idx))
except Exception as e: print('lfw failed',e)
try:
    z=zipfile.ZipFile(io.BytesIO(urllib.request.urlopen('https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip',timeout=120).read()))
    for nme in z.namelist():
        if nme.lower().endswith('.png') and '/PNGImages/' in nme:
            Image.open(io.BytesIO(z.read(nme))).convert('RGB').save(f"{out}/pf_{os.path.basename(nme)[:-4]}.jpg",quality=95); n+=1
    print('pennfudan ok')
except Exception as e: print('pennfudan failed',e)
try:
    from skimage import data
    for nm in ['astronaut','coffee','chelsea','rocket','camera']:
        pass
except Exception: pass
print('people photos:',n)
