const http=require('http'),fs=require('fs'),path=require('path');const {chromium}=require('playwright');
const root=process.cwd(),types={'.html':'text/html','.js':'text/javascript','.wasm':'application/wasm','.json':'application/json','.png':'image/png','.onnx':'application/octet-stream'};
const srv=http.createServer((q,r)=>{const p=path.join(root,decodeURIComponent(q.url.split('?')[0]));if(!p.startsWith(root)||!fs.existsSync(p)||fs.statSync(p).isDirectory()){r.writeHead(404);return r.end()}r.writeHead(200,{'Content-Type':types[path.extname(p)]||'application/octet-stream'});fs.createReadStream(p).pipe(r)}).listen(8123);
(async()=>{let report={ok:false};
 try{const b=await chromium.launch();const pg=await b.newPage();const logs=[];pg.on('console',m=>logs.push(m.text()));pg.on('pageerror',e=>logs.push('PAGEERR '+e));
  await pg.goto('http://localhost:8123/ci/test.html');
  const res=await pg.evaluate(()=>window.runTest());report={ok:true,log:res.log,console:logs.slice(-10)};
  fs.writeFileSync('report/browser_result.png',Buffer.from(res.png.split(',')[1],'base64'));await b.close();
 }catch(e){report={ok:false,error:String(e)}}
 fs.writeFileSync('report/browser_report.json',JSON.stringify(report,null,1));console.log(report);srv.close();})();
