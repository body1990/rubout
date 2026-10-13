import json, numpy as np, onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from skimage import data
sess = ort.InferenceSession('model/migan_pipeline_v2.onnx', providers=['CPUExecutionProvider'])
io = {'inputs': [], 'outputs': []}
for i in sess.get_inputs(): io['inputs'].append({'name': i.name, 'shape': [s if isinstance(s, int) else str(s) for s in i.shape], 'type': i.type})
for o in sess.get_outputs(): io['outputs'].append({'name': o.name, 'shape': [s if isinstance(s, int) else str(s) for s in o.shape], 'type': o.type})
print(json.dumps(io, indent=1))
def layout(shape):
    if len(shape) == 4: return 'nchw' if shape[1] in (1, 3) else 'nhwc'
    if len(shape) == 3: return 'chw' if shape[0] in (1, 3) else 'hwc'
    return 'hw'
io['img_layout'] = layout(io['inputs'][0]['shape']); io['mask_layout'] = layout(io['inputs'][1]['shape']); io['out_layout'] = layout(io['outputs'][0]['shape'])
io['img_input'] = io['inputs'][0]['name']; io['mask_input'] = io['inputs'][1]['name']
json.dump(io, open('report/migan_io.json', 'w'), indent=1)
# test image: portrait with a tiled text watermark
im = Image.fromarray(data.astronaut()).convert('RGB')  # 512x512
W, H = im.size
wm = Image.new('L', (W, H), 0); d = ImageDraw.Draw(wm)
try: f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 34)
except Exception: f = ImageFont.load_default()
txt = Image.new('L', (W * 2, H * 2), 0); td = ImageDraw.Draw(txt)
for y in range(0, H * 2, 110):
    for x in range(0, W * 2, 230): td.text((x, y), 'SAMPLE', font=f, fill=255)
txt = txt.rotate(28).crop((W // 2, H // 2, W // 2 + W, H // 2 + H))
a = np.asarray(txt, np.float32) / 255 * 0.55
marked = (np.asarray(im, np.float32) * (1 - a[..., None]) + 255 * a[..., None]).clip(0, 255).astype(np.uint8)
mask = (np.asarray(txt) > 30)
mask = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(5))) > 0
Image.fromarray(marked).save('report/test_in.png'); Image.fromarray((~mask * 255).astype(np.uint8)).save('report/test_mask.png')  # 255 = known
def prep(arr, lay):
    a = arr if arr.ndim == 3 else arr[..., None]
    if lay == 'nchw': return a.transpose(2, 0, 1)[None]
    if lay == 'chw': return a.transpose(2, 0, 1)
    if lay == 'hwc': return a
    if lay == 'hw': return arr
    return a[None]
mk = (~mask * 255).astype(np.uint8)
out = sess.run(None, {io['img_input']: prep(marked, io['img_layout']), io['mask_input']: prep(mk, io['mask_layout'])})[0]
print('out', out.shape, out.dtype)
o = {'nchw': lambda: out[0].transpose(1, 2, 0), 'chw': lambda: out.transpose(1, 2, 0), 'hwc': lambda: out, 'nhwc': lambda: out[0]}[io['out_layout']]()
o = o.astype(np.uint8)
side = np.concatenate([np.asarray(im), marked, o], 1)
Image.fromarray(side).save('report/py_result.png')
err = np.abs(o.astype(int) - np.asarray(im).astype(int)).mean(); print('mean abs err vs clean', err)
json.dump({'mean_abs_err_vs_clean': float(err)}, open('report/py_metrics.json', 'w'))
