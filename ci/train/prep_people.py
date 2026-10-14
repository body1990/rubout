# Collect people photos (faces + full-body pedestrians) from public datasets into /tmp/people/
import os, io, zipfile, urllib.request, numpy as np
from PIL import Image
out='/tmp/people'; os.makedirs(out, exist_ok=True); n=0
def _save_faces(arr_iter, limit=2500):
    k = 0
    for a in arr_iter:
        a = np.asarray(a)
        if a.dtype != np.uint8: a = (a * 255 if a.max() <= 1.5 else a).clip(0, 255).astype(np.uint8)
        Image.fromarray(a).save(f'{out}/lfw_{k}.jpg', quality=95); k += 1
        if k >= limit: break
    return k
k = 0
try:  # source 1: the original LFW archive
    import tarfile, urllib.request
    urllib.request.urlretrieve('http://vis-www.cs.umass.edu/lfw/lfw.tgz', '/tmp/lfw.tgz')
    tf = tarfile.open('/tmp/lfw.tgz')
    def it():
        for i, mem in enumerate(tf):
            if mem.isfile() and mem.name.lower().endswith('.jpg') and i % 5 == 0: yield Image.open(tf.extractfile(mem)).convert('RGB')
    k = _save_faces((np.asarray(x) for x in it()))
    print('lfw (umass) ok', k)
except Exception as e: print('lfw umass failed', e)
if k == 0:
    try:  # source 2: scikit-learn's LFW mirror (figshare); keep memory small
        from sklearn.datasets import fetch_lfw_people
        d = fetch_lfw_people(color=True, slice_=(slice(0, 250), slice(0, 250)), resize=1.0, min_faces_per_person=25, download_if_missing=True)
        idx = np.random.RandomState(0).permutation(len(d.images))
        k = _save_faces((d.images[i] for i in idx))
        print('lfw (sklearn) ok', k)
    except Exception as e: print('lfw sklearn failed', e)
n += k
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
