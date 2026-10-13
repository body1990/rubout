# Collect people photos (faces + full-body pedestrians) from public datasets into /tmp/people/
import os, io, zipfile, urllib.request, numpy as np
from PIL import Image
out='/tmp/people'; os.makedirs(out, exist_ok=True); n=0
try:
    import tarfile, urllib.request
    urllib.request.urlretrieve('http://vis-www.cs.umass.edu/lfw/lfw.tgz','/tmp/lfw.tgz')
    tf=tarfile.open('/tmp/lfw.tgz'); k=0
    for i,mem in enumerate(tf):
        if mem.isfile() and mem.name.lower().endswith('.jpg') and i%5==0:
            Image.open(tf.extractfile(mem)).convert('RGB').save(f'{out}/lfw_{k}.jpg',quality=95); k+=1; n+=1
            if k>=2500: break
    print('lfw ok',k)
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
