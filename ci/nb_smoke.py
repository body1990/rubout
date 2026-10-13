import json,re,sys
nb=json.load(open('train/rubout_train_gpu.ipynb')); out=[]
for c in nb['cells']:
    if c['cell_type']!='code': continue
    s=''.join(c['source'])
    if 'google.colab' in s: continue
    s=s.replace("sh('pip -q install","pass #sh('pip -q install").replace("sh('apt-get","pass #sh('apt-get").replace("sh('python rubout/ci/train/prep_people.py","pass #sh('python rubout/ci/train/prep_people.py")
    s=s.replace("if not os.path.exists('rubout'): sh('git clone --depth 1 https://github.com/body1990/rubout rubout')","pass")
    s=s.replace("assert dev == 'cuda', 'Turn on a GPU runtime first (Runtime > Change runtime type).'","pass")
    s=s.replace("MAX_MINUTES = 75","MAX_MINUTES = 3").replace("BATCH = 24","BATCH = 8").replace("CROP = 256","CROP = 128").replace("range(600)","range(60)").replace("step % 400","step % 25")
    s=s.replace("torch.autocast('cuda', dtype=torch.float16)","torch.autocast('cpu', dtype=torch.bfloat16)")
    s=s.replace("torch.cuda.amp.GradScaler()","torch.amp.GradScaler('cpu', enabled=False)")
    s=s.replace("nw = min(8, os.cpu_count() or 2)","nw = 2")
    out.append(s)
open('/tmp/nb_run.py','w').write('\n'.join(out))
