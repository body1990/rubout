import jax, jax.numpy as jnp, numpy as np, time, sys, json, pickle, os
sys.path.insert(0,os.path.dirname(__file__))
from train_old import init, net, prep, loss_fn, predict, evaluate, export
@jax.jit
def step(p,m,v,t,lr,x,y):
    l,g=jax.value_and_grad(loss_fn)(p,x,y); b1,b2=0.9,0.999
    m=jax.tree_util.tree_map(lambda m,g:b1*m+(1-b1)*g,m,g); v=jax.tree_util.tree_map(lambda v,g:b2*v+(1-b2)*g*g,v,g)
    p=jax.tree_util.tree_map(lambda p,m,v:p-lr*(m/(1-b1**t))/(jnp.sqrt(v/(1-b2**t))+1e-8),p,m,v); return p,m,v,l
if __name__=='__main__':
    budget=float(sys.argv[1])*60; out=sys.argv[2]; bs=int(sys.argv[3]) if len(sys.argv)>3 else 16; lr0=1.5e-3
    d=np.load('/tmp/train_data.npz'); X,Y=d['X'],d['Y']; dv=np.load('/tmp/val_data.npz'); Xv,Yv=dv['X'],dv['Y']
    print('train',X.shape,'val',Xv.shape,flush=True)
    p=init(jax.random.PRNGKey(0))
    if os.path.exists('/tmp/init.pkl'): p=jax.tree_util.tree_map(jnp.asarray,pickle.load(open('/tmp/init.pkl','rb'))); print('warm start',flush=True)
    m=jax.tree_util.tree_map(jnp.zeros_like,p); v=jax.tree_util.tree_map(jnp.zeros_like,p)
    rng=np.random.RandomState(0); t0=time.time(); t=0; best=(-1,None)
    while True:
        el=time.time()-t0; frac=min(1.0,el/budget)
        idx=rng.randint(0,len(X),bs); xb,yb=X[idx],Y[idx]; k=rng.randint(4)
        if k: xb=np.rot90(xb,k,(1,2)); yb=np.rot90(yb,k,(1,2))
        if rng.rand()<0.5: xb=xb[:,:,::-1]; yb=yb[:,:,::-1]
        t+=1; lr=lr0*min(1,t/100)*(0.02+0.98*0.5*(1+np.cos(np.pi*frac)))
        p,m,v,l=step(p,m,v,float(t),lr,np.ascontiguousarray(xb),np.ascontiguousarray(yb))
        if t%500==0:
            iou,rec,prec=evaluate(p,Xv,Yv); print(f't={t} loss={float(l):.4f} lr={lr:.5f} val IoU={iou:.3f} rec={rec:.3f} prec={prec:.3f} elapsed={el/60:.1f}m',flush=True)
            if iou>best[0] and frac>0.5: best=(iou,jax.tree_util.tree_map(np.asarray,p))
        if frac>=1: break
    iou,rec,prec=evaluate(p,Xv,Yv); print('FINAL val IoU %.3f rec %.3f prec %.3f'%(iou,rec,prec),flush=True)
    if best[1] is not None and best[0]>iou: p=jax.tree_util.tree_map(jnp.asarray,best[1]); print('using best checkpoint',best[0])
    pickle.dump(jax.tree_util.tree_map(np.asarray,p),open(out+'.pkl','wb')); export(p,out+'_export'); print('exported',flush=True)
