import jax, jax.numpy as jnp, numpy as np, time, sys, json, pickle, os
SPEC=[('e1a',3,8),('e1b',8,8),('e2a',8,16),('e2b',16,16),('e3a',16,32),('e3b',32,32),('bna',32,48),('bnb',48,48),
      ('d3a',48+32,32),('d3b',32,32),('d2a',32+16,16),('d2b',16,16),('d1a',16+8,8),('d1b',8,8)]
def init(key):
    p={}
    for name,ci,co in SPEC:
        key,k=jax.random.split(key); p[name]={'w':jax.random.normal(k,(3,3,ci,co))*np.sqrt(2.0/(9*ci)),'b':jnp.zeros(co)}
    key,k=jax.random.split(key); p['out']={'w':jax.random.normal(k,(1,1,8,1))*np.sqrt(1.0/8),'b':jnp.full((1,),-2.0)}
    return p
def conv(p,x,relu=True):
    y=jax.lax.conv_general_dilated(x,p['w'],(1,1),'SAME',dimension_numbers=('NHWC','HWIO','NHWC'))+p['b']
    return jax.nn.relu(y) if relu else y
def pool(x): return jax.lax.reduce_window(x,-jnp.inf,jax.lax.max,(1,2,2,1),(1,2,2,1),'VALID')
def up(x): return jnp.repeat(jnp.repeat(x,2,1),2,2)
def net(p,x):
    e1=conv(p['e1b'],conv(p['e1a'],x)); e2=conv(p['e2b'],conv(p['e2a'],pool(e1))); e3=conv(p['e3b'],conv(p['e3a'],pool(e2)))
    b=conv(p['bnb'],conv(p['bna'],pool(e3)))
    d3=conv(p['d3b'],conv(p['d3a'],jnp.concatenate([up(b),e3],-1)))
    d2=conv(p['d2b'],conv(p['d2a'],jnp.concatenate([up(d3),e2],-1)))
    d1=conv(p['d1b'],conv(p['d1a'],jnp.concatenate([up(d2),e1],-1)))
    return conv(p['out'],d1,False)[...,0]
def prep(xu8): return xu8.astype(jnp.float32)/255.0-0.5
def loss_fn(p,x,y):
    lg=net(p,prep(x)); y=y.astype(jnp.float32)
    bce=jnp.mean(jnp.maximum(lg,0)-lg*y+jnp.log1p(jnp.exp(-jnp.abs(lg))))
    pr=jax.nn.sigmoid(lg); inter=(pr*y).sum((1,2)); dice=1-(2*inter+1)/(pr.sum((1,2))+y.sum((1,2))+1)
    return bce+0.5*dice.mean()
@jax.jit
def step(p,m,v,t,lr,x,y):
    l,g=jax.value_and_grad(loss_fn)(p,x,y)
    b1,b2=0.9,0.999
    m=jax.tree_util.tree_map(lambda m,g:b1*m+(1-b1)*g,m,g); v=jax.tree_util.tree_map(lambda v,g:b2*v+(1-b2)*g*g,v,g)
    p=jax.tree_util.tree_map(lambda p,m,v:p-lr*(m/(1-b1**t))/(jnp.sqrt(v/(1-b2**t))+1e-8),p,m,v)
    return p,m,v,l
@jax.jit
def predict(p,x): return jax.nn.sigmoid(net(p,prep(x)))
def evaluate(p,X,Y,bs=50):
    tp=fp=fn=0
    for i in range(0,len(X),bs):
        pr=np.asarray(predict(p,X[i:i+bs]))>0.5; y=Y[i:i+bs]>0
        tp+=(pr&y).sum(); fp+=(pr&~y).sum(); fn+=(~pr&y).sum()
    return tp/max(1,tp+fp+fn), tp/max(1,tp+fn), tp/max(1,tp+fp)
def export(p,outdir):
    os.makedirs(outdir,exist_ok=True); blobs=[]; meta=[]; off=0
    for name,ci,co in SPEC+[('out',8,1)]:
        w=np.asarray(p[name]['w'],np.float32); b=np.asarray(p[name]['b'],np.float32); k=w.shape[0]
        meta.append({'name':name,'k':k,'cin':ci,'cout':co,'woff':off,'boff':off+w.size}); blobs+= [w.ravel(),b.ravel()]; off+=w.size+b.size
    np.concatenate(blobs).astype('<f4').tofile(outdir+'/weights.bin'); json.dump(meta,open(outdir+'/model.json','w'))
if __name__=='__main__':
    budget=float(sys.argv[1])*60; out=sys.argv[2]; bs=16
    d=np.load('/tmp/train_data.npz'); X,Y=d['X'],d['Y']; dv=np.load('/tmp/val_data.npz'); Xv,Yv=dv['X'],dv['Y']
    print('train',X.shape,'val',Xv.shape,flush=True)
    p=init(jax.random.PRNGKey(0)); m=jax.tree_util.tree_map(jnp.zeros_like,p); v=jax.tree_util.tree_map(jnp.zeros_like,p)
    rng=np.random.RandomState(0); t0=time.time(); total=None; t=0; lr0=3e-3
    while True:
        idx=rng.randint(0,len(X),bs); xb,yb=X[idx],Y[idx]
        k=rng.randint(4)
        if k: xb=np.rot90(xb,k,(1,2)); yb=np.rot90(yb,k,(1,2))
        if rng.rand()<0.5: xb=xb[:,:,::-1]; yb=yb[:,:,::-1]
        t+=1
        if t==30: total=int(budget/((time.time()-t0)/30)); print('est total steps',total,flush=True)
        frac=min(1.0,t/total) if total else 0
        lr=lr0*min(1,t/50)*(0.03+0.97*0.5*(1+np.cos(np.pi*frac)))
        p,m,v,l=step(p,m,v,float(t),lr,np.ascontiguousarray(xb),np.ascontiguousarray(yb))
        if t%100==0:
            iou,rec,prec=evaluate(p,Xv,Yv); print(f't={t} loss={float(l):.4f} lr={lr:.5f} val IoU={iou:.3f} rec={rec:.3f} prec={prec:.3f} elapsed={(time.time()-t0)/60:.1f}m',flush=True)
            pickle.dump(jax.tree_util.tree_map(np.asarray,p),open(out+'.pkl','wb'))
        if total and t>=total: break
    iou,rec,prec=evaluate(p,Xv,Yv); print('FINAL val IoU %.3f rec %.3f prec %.3f'%(iou,rec,prec),flush=True)
    pickle.dump(jax.tree_util.tree_map(np.asarray,p),open(out+'.pkl','wb')); export(p,out+'_export'); print('exported',flush=True)
