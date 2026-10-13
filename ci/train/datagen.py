import numpy as np, os, random, io, sys, subprocess, math
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
PHOTOS=os.environ.get('PHOTOS','/tmp/photos/')
EXTRA=[d for d in os.environ.get('EXTRA','').split(':') if d]
def list_fonts():
    out=subprocess.run(['fc-list','--format','%{file}|%{family}|%{style}\n'],capture_output=True,text=True).stdout.split('\n')
    bad=('math','symbol','dingbat','emoji','wingding','ipa','cjk','noto sans','lohit','kacst','tlwg','droid sans fallback','nanum','unifont','fontawesome','fira code','phetsarith','sinhala','khmer','thai','lao','tibetan','mongolian','ethiopic','malayalam','tamil','telugu','gujarati','kannada','bengali','devanagari','oriya','gurmukhi','myanmar','arabic','hebrew','armenian','georgian')
    seen={}; res=[]
    for l in out:
        p=l.split('|')
        if len(p)<3: continue
        f,fam,sty=p[0],p[1].lower(),p[2].lower()
        if not f.lower().endswith(('.ttf','.otf')): continue
        if any(b in fam for b in bad): continue
        k=(fam.split(',')[0],sty.split(',')[0])
        if k in seen: continue
        seen[k]=1
        try:
            ft=ImageFont.truetype(f,40); bb=ft.getbbox('Ag12'); 
            if bb[2]-bb[0]<20: continue
            res.append(f)
        except Exception: pass
    return res
FONTS=list_fonts()
MONO=[f for f in FONTS if 'mono' in f.lower() or 'courier' in f.lower()] or FONTS
SYL=['ar','bel','cor','dex','el','fo','gra','hy','im','jo','ka','lu','mon','nor','ov','pix','qua','ri','sol','tra','un','vi','wen','xo','yu','zen','stock','photo','image','snap','lens','frame','studio','art','pic','shot','hub','box','lab']
WORDS=['SAMPLE','PREVIEW','DRAFT','CONFIDENTIAL','COPYRIGHT','PROOF','DEMO','WATERMARK','NOT FOR SALE','Property of','All rights reserved','Photography','STOCK','Studio','Images','Photo','Official','Do not copy','PRIVATE','@username']
def brand(rng):
    t=rng.random()
    if t<0.35: s=''.join(rng.choice(SYL) for _ in range(rng.randint(2,3))); s=s.capitalize() if rng.random()<0.6 else s.upper()
    elif t<0.6: s=rng.choice(WORDS)
    elif t<0.75: s='www.'+''.join(rng.choice(SYL) for _ in range(2))+'.com'
    elif t<0.85: s='@'+''.join(rng.choice(SYL) for _ in range(2))
    else: s=rng.choice(WORDS)+' '+''.join(rng.choice(SYL) for _ in range(2)).capitalize()
    if rng.random()<0.3: s='© '+s
    if rng.random()<0.12: s+=' '+str(rng.randint(2012,2026))
    return s
def font(path,size): return ImageFont.truetype(path,max(6,int(size)))
def color(rng,kind=None):
    k=kind or rng.choices(['w','k','g','c'],[0.45,0.25,0.1,0.2])[0]
    if k=='w': return (255,255,255)
    if k=='k': return (0,0,0)
    if k=='g': v=rng.randint(60,200); return (v,v,v)
    import colorsys
    r,g,b=colorsys.hsv_to_rgb(rng.random(),rng.uniform(0.5,1),rng.uniform(0.7,1)); return (int(r*255),int(g*255),int(b*255))

# ---------- backgrounds ----------
PH=[PHOTOS+f for f in sorted(os.listdir(PHOTOS))]
PEOPLE=[d+'/'+f for d in EXTRA for f in sorted(os.listdir(d))]
_cache={}
def load_photo(name):
    if name not in _cache:
        im=Image.open(name).convert('RGB'); s=900/max(im.size)
        if s<1: im=im.resize((int(im.width*s),int(im.height*s)),Image.BICUBIC)
        _cache[name]=im
    return _cache[name]
def noise_field(rng,W,H,octaves):
    acc=np.zeros((H,W),np.float32); amp=1.0
    for o in range(octaves):
        g=rng.randint(3,6)*(2**o); small=np.random.RandomState(rng.randint(0,1<<30)).rand(max(2,H*g//max(W,H)+2),g+2).astype(np.float32)
        im=Image.fromarray((small*255).astype(np.uint8)).resize((W,H),Image.BICUBIC); acc+=np.asarray(im,np.float32)/255*amp; amp*=0.55
    acc-=acc.min(); acc/=max(1e-6,acc.max()); return acc
def procedural(rng,W,H):
    pal=[np.array([rng.randint(0,255) for _ in range(3)],np.float32) for _ in range(3)]
    n1=noise_field(rng,W,H,rng.randint(2,5))[...,None]; n2=noise_field(rng,W,H,rng.randint(3,6))[...,None]
    a=pal[0]*(1-n1)+pal[1]*n1; a=a*(1-0.5*n2)+pal[2]*0.5*n2
    im=Image.fromarray(np.clip(a,0,255).astype(np.uint8)); d=ImageDraw.Draw(im)
    for _ in range(rng.randint(0,14)):
        c=tuple(rng.randint(0,255) for _ in range(3)); x0,y0=rng.randint(0,W),rng.randint(0,H); x1,y1=x0+rng.randint(10,W//2),y0+rng.randint(10,H//2)
        t=rng.random()
        if t<0.4: d.rectangle([x0,y0,x1,y1],fill=c)
        elif t<0.8: d.ellipse([x0,y0,x1,y1],fill=c)
        else: d.polygon([(x0,y0),(x1,y0+rng.randint(0,40)),(x0+rng.randint(0,60),y1)],fill=c)
    if rng.random()<0.5: im=im.filter(ImageFilter.GaussianBlur(rng.uniform(0.5,3)))
    return im
def real_crop(rng,W,H,names):
    im=load_photo(rng.choice(PEOPLE) if (PEOPLE and rng.random()<0.65) else rng.choice(names)); w,h=im.size
    f=rng.uniform(0.35,1.0); cw,ch=int(w*f),int(h*f*rng.uniform(0.8,1.2)); ch=min(ch,h); cw=min(cw,w)
    x,y=rng.randint(0,w-cw),rng.randint(0,h-ch); im=im.crop((x,y,x+cw,y+ch))
    if rng.random()<0.5: im=im.transpose(Image.FLIP_LEFT_RIGHT)
    return im.resize((W,H),Image.BICUBIC)
def background(rng,names):
    s=rng.uniform(420,640); asp=rng.choice([1.0,4/3,3/2,16/9,3/4,2/3,1.2])
    W,H=(int(s),int(s/asp)) if asp>=1 else (int(s*asp),int(s))
    t=rng.random()
    if t<0.62: im=real_crop(rng,W,H,names)
    elif t<0.85: im=procedural(rng,W,H)
    else:
        a=real_crop(rng,W,H,names); b=real_crop(rng,W,H,names) if rng.random()<0.7 else procedural(rng,W,H)
        m=Image.new('L',(W,H),0); d=ImageDraw.Draw(m)
        for _ in range(rng.randint(1,3)):
            x0,y0=rng.randint(0,W),rng.randint(0,H); x1,y1=rng.randint(0,W),rng.randint(0,H)
            if rng.random()<0.5: d.rectangle([min(x0,x1),min(y0,y1),max(x0,x1),max(y0,y1)],fill=255)
            else: d.polygon([(x0,y0),(x1,y1),(rng.randint(0,W),rng.randint(0,H))],fill=255)
        im=Image.composite(b,a,m)
    if rng.random()<0.7:
        im=ImageEnhance.Brightness(im).enhance(rng.uniform(0.6,1.35)); im=ImageEnhance.Contrast(im).enhance(rng.uniform(0.6,1.4)); im=ImageEnhance.Color(im).enhance(rng.uniform(0.3,1.5))
    if rng.random()<0.06: im=im.convert('L').convert('RGB')
    arr=np.asarray(im,np.float32)
    if rng.random()<0.6: arr=arr+np.random.RandomState(rng.randint(0,1<<30)).normal(0,rng.uniform(0.5,4),arr.shape[:2]+(1,))
    return np.clip(arr,0,255)

# ---------- watermark layers (return alpha HxW in 0..1 and colour) ----------
def text_canvas(size):
    return Image.new('L',size,0)
def tiled(rng,W,H):
    fp=rng.choice(FONTS); fs=min(W,H)*rng.uniform(0.045,0.17); ft=font(fp,fs); txt=brand(rng)
    D=int(math.hypot(W,H))+20; cv=text_canvas((D,D)); d=ImageDraw.Draw(cv)
    tw=d.textlength(txt,font=ft); sx=tw*rng.uniform(1.15,2.2); sy=fs*rng.uniform(1.8,5.0); st=rng.random()<0.6
    sw=rng.choice([0,0,0,max(1,int(fs/18))]) if rng.random()<0.25 else 0
    y=-sy; r=0
    while y<D+sy:
        x=-sx+(sx/2 if (st and r%2) else 0)+rng.uniform(0,sx)*0
        while x<D+sx:
            d.text((x,y),txt,font=ft,fill=255,anchor='lm',stroke_width=sw); x+=sx
        y+=sy; r+=1
    ang=rng.choice([0,0,rng.uniform(-60,60),rng.uniform(-60,60),30,-30,45,-45]); cv=cv.rotate(ang,resample=Image.BICUBIC,center=(D/2,D/2))
    cv=cv.crop(((D-W)//2,(D-H)//2,(D-W)//2+W,(D-H)//2+H)); return np.asarray(cv,np.float32)/255
def centered(rng,W,H):
    fp=rng.choice(FONTS); fs=min(W,H)*rng.uniform(0.1,0.3); ft=font(fp,fs); lines=[brand(rng)] if rng.random()<0.7 else [brand(rng),brand(rng)]
    cv=text_canvas((W*2,H*2)); d=ImageDraw.Draw(cv); y=H-(len(lines)-1)*fs*0.6
    for l in lines: d.text((W,y),l,font=ft,fill=255,anchor='mm'); y+=fs*1.2
    ang=rng.choice([0,rng.uniform(-45,45),-30,30,45,-45]); cv=cv.rotate(ang,resample=Image.BICUBIC,center=(W,H)); cv=cv.crop((W//2,H//2,W//2+W,H//2+H)); return np.asarray(cv,np.float32)/255
def icon(d,cx,cy,r,rng,w):
    t=rng.choice(['ring','disc','square','tri','star','hex'])
    if t=='ring': d.ellipse([cx-r,cy-r,cx+r,cy+r],outline=255,width=max(1,int(r*0.22)))
    elif t=='disc': d.ellipse([cx-r,cy-r,cx+r,cy+r],fill=255)
    elif t=='square': d.rounded_rectangle([cx-r,cy-r,cx+r,cy+r],radius=r*0.25,outline=255,width=max(1,int(r*0.22)))
    elif t=='tri': d.polygon([(cx,cy-r),(cx+r,cy+r),(cx-r,cy+r)],outline=255,width=max(1,int(r*0.2)))
    elif t=='star':
        pts=[(cx+(r if i%2==0 else r*0.45)*math.sin(i*math.pi/5),cy-(r if i%2==0 else r*0.45)*math.cos(i*math.pi/5)) for i in range(10)]; d.polygon(pts,fill=255)
    else: d.polygon([(cx+r*math.cos(i*math.pi/3),cy+r*math.sin(i*math.pi/3)) for i in range(6)],outline=255,width=max(1,int(r*0.22)))
def corner(rng,W,H):
    fp=rng.choice(FONTS); fs=min(W,H)*rng.uniform(0.035,0.09); ft=font(fp,fs); txt=brand(rng)
    cv=text_canvas((W,H)); d=ImageDraw.Draw(cv); m=min(W,H)*rng.uniform(0.02,0.06); right=rng.random()<0.65; bottom=rng.random()<0.75
    tw=d.textlength(txt,font=ft); ic=rng.random()<0.4; r=fs*0.55
    x=(W-m-tw) if right else m+(2.4*r if ic else 0); y=(H-m-fs*0.5) if bottom else m+fs*0.5
    d.text((x,y),txt,font=ft,fill=255,anchor='lm')
    if ic: icon(d,x-1.4*r,y,r,rng,fs)
    return np.asarray(cv,np.float32)/255
def logo(rng,W,H):
    fp=rng.choice(FONTS); R=min(W,H)*rng.uniform(0.06,0.14); cv=text_canvas((W,H)); d=ImageDraw.Draw(cv)
    cx,cy=rng.choice([(W/2,H/2),(rng.uniform(R*1.5,W-R*1.5),rng.uniform(R*1.5,H-R*1.5))])
    icon(d,cx,cy,R,rng,R); txt=rng.choice(['LOGO','STUDIO','PHOTO',brand(rng)[:10],'©','S','P'])
    d.text((cx,cy+(R*1.5 if rng.random()<0.4 else 0)),txt,font=font(fp,R*rng.uniform(0.5,0.9)),fill=255,anchor='mm')
    return np.asarray(cv,np.float32)/255
def datestamp(rng,W,H):
    fp=rng.choice(MONO); fs=min(W,H)*rng.uniform(0.04,0.07); y,mo,dd=rng.randint(2005,2026),rng.randint(1,12),rng.randint(1,28)
    txt=rng.choice([f"'{y%100:02d} {mo} {dd}",f"{y}.{mo:02d}.{dd:02d}",f"{dd:02d}/{mo:02d}/{y}",f"{mo:02d}-{dd:02d}-{y} {rng.randint(0,23):02d}:{rng.randint(0,59):02d}"])
    cv=text_canvas((W,H)); d=ImageDraw.Draw(cv); m=min(W,H)*0.04; right=rng.random()<0.8
    d.text(((W-m) if right else m,H-m),txt,font=font(fp,fs),fill=255,anchor='rb' if right else 'lb'); return np.asarray(cv,np.float32)/255
def band(rng,W,H):
    cv=text_canvas((W,H)); d=ImageDraw.Draw(cv); bh=int(H*rng.uniform(0.07,0.16)); top=rng.random()<0.25; y0=0 if top else H-bh
    d.rectangle([0,y0,W,y0+bh],fill=255); return np.asarray(cv,np.float32)/255, (y0,bh)
def lines(rng,W,H):
    cv=text_canvas((W,H)); d=ImageDraw.Draw(cv); w=max(1,int(min(W,H)*rng.uniform(0.004,0.02)))
    if rng.random()<0.5: d.line([(0,0),(W,H)],fill=255,width=w); d.line([(0,H),(W,0)],fill=255,width=w)
    else:
        for i in range(rng.randint(2,6)): o=rng.randint(-H,W); d.line([(o,0),(o+H,H)],fill=255,width=w)
    return np.asarray(cv,np.float32)/255
def make_layers(rng,W,H):
    layers=[]; kinds=rng.choices(['tiled','centered','corner','logo','date','band','lines'],[0.28,0.14,0.2,0.1,0.1,0.08,0.04],k=2 if rng.random()<0.25 else 1)
    for k in kinds:
        if k=='band':
            m,(y0,bh)=band(rng,W,H); a=m*rng.uniform(0.2,0.6); col=color(rng,rng.choice(['k','k','w','c'])); layers.append((a,col))
            cv=text_canvas((W,H)); d=ImageDraw.Draw(cv); d.text((W/2,y0+bh/2),brand(rng),font=font(rng.choice(FONTS),bh*0.55),fill=255,anchor='mm')
            layers.append((np.asarray(cv,np.float32)/255*rng.uniform(0.7,1.0),color(rng,rng.choice(['w','k']))))
            continue
        fn={'tiled':tiled,'centered':centered,'corner':corner,'logo':logo,'date':datestamp,'lines':lines}[k]; m=fn(rng,W,H)
        if k=='date': col=rng.choice([(255,140,20),(255,200,40),(255,90,30),(255,255,255),(255,160,60)]); a=m*rng.uniform(0.85,1.0)
        elif k=='corner': col=color(rng); a=m*rng.choice([rng.uniform(0.25,0.7),rng.uniform(0.7,1.0)])
        else: col=color(rng); a=m*rng.uniform(0.08,0.7)
        if rng.random()<0.15 and k in('tiled','centered','corner'):   # soft shadow
            sh=np.roll(np.roll(a,2,0),2,1)*0.8; layers.append((sh,(0,0,0)))
        layers.append((a,col))
    return layers
def sample(rng,names,crop):
    bg=background(rng,names); H,W=bg.shape[:2]; truth=np.zeros((H,W),bool); img=bg
    if rng.random()>0.18:
        for a,col in make_layers(rng,W,H):
            a=np.clip(a,0,1); img=img*(1-a[...,None])+np.array(col,np.float32)*a[...,None]; truth|=a>0.03
    img=np.clip(img+0.5,0,255).astype(np.uint8); im=Image.fromarray(img)
    if rng.random()<0.15: im=im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3,0.8)))
    if rng.random()<0.6:
        b=io.BytesIO(); im.save(b,'JPEG',quality=rng.randint(55,95)); b.seek(0); im=Image.open(b).convert('RGB')
    img=np.asarray(im); 
    if rng.random()<0.7 and truth.any():
        ys,xs=np.nonzero(truth); k=rng.randrange(len(ys)); cy,cx=ys[k],xs[k]
        y0=int(np.clip(cy-rng.randint(0,crop-1),0,max(0,H-crop))); x0=int(np.clip(cx-rng.randint(0,crop-1),0,max(0,W-crop)))
    else: y0,x0=rng.randint(0,max(0,H-crop)),rng.randint(0,max(0,W-crop))
    X=img[y0:y0+crop,x0:x0+crop]; Y=truth[y0:y0+crop,x0:x0+crop]
    if X.shape[0]<crop or X.shape[1]<crop:
        X=np.pad(X,((0,crop-X.shape[0]),(0,crop-X.shape[1]),(0,0)),mode='edge'); Y=np.pad(Y,((0,crop-Y.shape[0]),(0,crop-Y.shape[1])))
    return X,Y.astype(np.uint8)
def work(args):
    seed,n,crop,names=args; rng=random.Random(seed); np.random.seed(seed%(2**31))
    X=np.zeros((n,crop,crop,3),np.uint8); Y=np.zeros((n,crop,crop),np.uint8)
    for i in range(n): X[i],Y[i]=sample(rng,names,crop)
    return X,Y
if __name__=='__main__':
    from multiprocessing import Pool
    out,total,crop,seed0=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]); per=100
    names=PH
    with Pool(int(os.environ.get('NPROC','2'))) as p: res=p.map(work,[(seed0+i,per,crop,names) for i in range(total//per)])
    X=np.concatenate([r[0] for r in res]); Y=np.concatenate([r[1] for r in res]); np.savez(out,X=X,Y=Y); print(out,X.shape,'mask frac',Y.mean(),'fonts',len(FONTS))
