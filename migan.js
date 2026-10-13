/* Run the MI-GAN inpainting pipeline (ONNX) through onnxruntime. Shared by the app and the CI test.
   rgba: Uint8ClampedArray (w*h*4), hole: Uint8Array (w*h, 1 = remove). Returns a new RGBA array. */
function miganLayout(shape) {
  if (shape.length === 4) return (shape[1] === 1 || shape[1] === 3) ? 'nchw' : 'nhwc';
  if (shape.length === 3) return (shape[0] === 1 || shape[0] === 3) ? 'chw' : 'hwc';
  return 'hw';
}
function miganDims(lay, c, h, w) {
  return { nchw: [1, c, h, w], nhwc: [1, h, w, c], chw: [c, h, w], hwc: [h, w, c], hw: [h, w] }[lay];
}
async function miganRun(ort, session, io, rgba, hole, w, h) {
  var n = w * h, i, c;
  var il = miganLayout(io.inputs[0].shape), ml = miganLayout(io.inputs[1].shape);
  var img = new Uint8Array(n * 3), planar = (il === 'nchw' || il === 'chw');
  for (i = 0; i < n; i++) for (c = 0; c < 3; c++) img[planar ? c * n + i : i * 3 + c] = rgba[i * 4 + c];
  var mk = new Uint8Array(n); for (i = 0; i < n; i++) mk[i] = hole[i] ? 0 : 255; // 255 = keep, 0 = fill
  var feeds = {};
  feeds[io.img_input] = new ort.Tensor('uint8', img, miganDims(il, 3, h, w));
  feeds[io.mask_input] = new ort.Tensor('uint8', mk, miganDims(ml, 1, h, w));
  var res = await session.run(feeds), out = res[session.outputNames[0]], d = out.dims, data = out.data;
  var oh, ow, ol = miganLayout(d);
  if (ol === 'nchw') { oh = d[2]; ow = d[3]; } else if (ol === 'nhwc') { oh = d[1]; ow = d[2]; } else if (ol === 'chw') { oh = d[1]; ow = d[2]; } else { oh = d[0]; ow = d[1]; }
  var on = ow * oh, res2 = new Uint8ClampedArray(on * 4), op = (ol === 'nchw' || ol === 'chw');
  for (i = 0; i < on; i++) { for (c = 0; c < 3; c++) res2[i * 4 + c] = data[op ? c * on + i : i * 3 + c]; res2[i * 4 + 3] = 255; }
  return { rgba: res2, w: ow, h: oh };
}
if (typeof module !== 'undefined') module.exports = miganRun;
