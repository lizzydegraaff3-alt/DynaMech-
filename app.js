const OPS=['rho','mu','iotaPos','iotaNeg','C','D','cl','B'];let selected=null,data=null,detail=1,threshold=.6;const $=x=>document.getElementById(x),clamp=x=>Math.max(0,Math.min(1,x));
function norm(a){let lo=Infinity,hi=-Infinity;for(const v of a){if(v<lo)lo=v;if(v>hi)hi=v}const o=new Float32Array(a.length);if(hi<=lo+1e-9)return o;for(let i=0;i<a.length;i++)o[i]=clamp((a[i]-lo)/(hi-lo));return o}
function idx(x,y,w,h){return Math.max(0,Math.min(h-1,y))*w+Math.max(0,Math.min(w-1,x))}function blur(a,w,h,r=1){let o=new Float32Array(a.length);for(let y=0;y<h;y++)for(let x=0;x<w;x++){let s=0,n=0;for(let j=-r;j<=r;j++)for(let i=-r;i<=r;i++){s+=a[idx(x+i,y+j,w,h)];n++}o[y*w+x]=s/n}return o}
function grad(a,w,h){let gx=new Float32Array(a.length),gy=new Float32Array(a.length),m=new Float32Array(a.length);for(let y=0;y<h;y++)for(let x=0;x<w;x++){let k=y*w+x;gx[k]=(a[idx(x+1,y,w,h)]-a[idx(x-1,y,w,h)])/2;gy[k]=(a[idx(x,y+1,w,h)]-a[idx(x,y-1,w,h)])/2;m[k]=Math.hypot(gx[k],gy[k])}return{gx,gy,m}}
function avg(xs){let o=new Float32Array(xs[0].length);for(const a of xs)for(let i=0;i<o.length;i++)o[i]+=a[i]/xs.length;return o}function mul(a,b){let o=new Float32Array(a.length);for(let i=0;i<o.length;i++)o[i]=a[i]*b[i];return norm(o)}
async function load(src,max=512){const blob=src instanceof File?src:await fetch(src).then(r=>r.blob());const bit=await createImageBitmap(blob),s=Math.min(1,max/Math.max(bit.width,bit.height)),w=Math.max(8,Math.round(bit.width*s)),h=Math.max(8,Math.round(bit.height*s)),c=document.createElement('canvas');c.width=w;c.height=h;const x=c.getContext('2d');x.drawImage(bit,0,0,w,h);const im=x.getImageData(0,0,w,h),L=new Float32Array(w*h);for(let p=0,k=0;p<im.data.length;p+=4,k++){let f=v=>{v/=255;return v<=.04045?v/12.92:((v+.055)/1.055)**2.4};L[k]=.2126*f(im.data[p])+.7152*f(im.data[p+1])+.0722*f(im.data[p+2])}return{c,w,h,L}}
function analyze(L,w,h){const g=grad(L,w,h),B=norm(blur(g.m,w,h,1)),mean=blur(B,w,h,3),sq=blur(Float32Array.from(B,(v,i)=>v*v),w,h,3),std=Float32Array.from(B,(v,i)=>Math.sqrt(Math.max(0,sq[i]-mean[i]*mean[i]))),rho=norm(Float32Array.from(std,v=>1-v)),mu=norm(std),ip=new Float32Array(L.length),im=new Float32Array(L.length),C=new Float32Array(L.length),D=new Float32Array(L.length);for(let y=0;y<h;y++)for(let x=0;x<w;x++){let k=y*w+x,lap=(g.gx[idx(x+1,y,w,h)]-g.gx[idx(x-1,y,w,h)]+g.gy[idx(x,y+1,w,h)]-g.gy[idx(x,y-1,w,h)])/2;ip[k]=Math.max(0,lap);im[k]=Math.max(0,-lap);let ax=Math.abs(g.gx[k]),ay=Math.abs(g.gy[k]);C[k]=Math.abs(ax-ay)/(ax+ay+1e-6);D[k]=(Math.atan2(g.gy[k],g.gx[k])+Math.PI)/(2*Math.PI)}const iotaPos=norm(ip),iotaNeg=norm(im),clRaw=Float32Array.from(B,(v,i)=>v*C[i]*(1-.5*iotaNeg[i])),cl=norm(clRaw),G={rho,mu,iotaPos,iotaNeg,C:norm(C),D,cl,B};const positive=avg([rho,mu,iotaPos,G.C,cl,B]),negative=avg([iotaNeg,D]),fgRaw=Float32Array.from(positive,(v,i)=>v*(1-.5*negative[i]));let FG=norm(fgRaw);FG=norm(Float32Array.from(FG,(v,i)=>v>Math.max(.12,.28/detail)?v:0));const LK={};for(const op of OPS)LK[op]=mul(G[op],L);const lp=avg(['rho','mu','iotaPos','C','cl','B'].map(x=>LK[x])),ln=avg(['iotaNeg','D'].map(x=>LK[x])),LOP=norm(Float32Array.from(lp,(v,i)=>v*(1-.5*ln[i]))),FS=Float32Array.from(LOP,v=>(1-v)>=threshold?1:0);const contours=norm(g.m),primitives=detectPrimitives(FG,contours,w,h);return{G,LK,FG,contours,LOP,FS,primitives}}
function connectedComponents(mask,w,h,minPixels){const seen=new Uint8Array(mask.length),components=[];const qx=new Int32Array(mask.length),qy=new Int32Array(mask.length);for(let y=0;y<h;y++)for(let x=0;x<w;x++){const start=y*w+x;if(!mask[start]||seen[start])continue;let head=0,tail=0;qx[tail]=x;qy[tail++]=y;seen[start]=1;let minX=x,maxX=x,minY=y,maxY=y,count=0,sumX=0,sumY=0;while(head<tail){const cx=qx[head],cy=qy[head++];count++;sumX+=cx;sumY+=cy;minX=Math.min(minX,cx);maxX=Math.max(maxX,cx);minY=Math.min(minY,cy);maxY=Math.max(maxY,cy);for(const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]]){const nx=cx+dx,ny=cy+dy;if(nx<0||ny<0||nx>=w||ny>=h)continue;const ni=ny*w+nx;if(mask[ni]&&!seen[ni]){seen[ni]=1;qx[tail]=nx;qy[tail++]=ny}}}if(count>=minPixels)components.push({type:'component',x0:minX/w,y0:minY/h,x1:(maxX+1)/w,y1:(maxY+1)/h,cx:(sumX/count)/w,cy:(sumY/count)/h,support:count/(w*h)})}return components.sort((a,b)=>b.support-a.support).slice(0,8)}
function detectPrimitives(FG,contours,w,h){const row=new Float32Array(h),col=new Float32Array(w);let total=0;for(let y=0;y<h;y++)for(let x=0;x<w;x++){const v=FG[y*w+x];row[y]+=v;col[x]+=v;total+=v}const topRows=[...row.keys()].sort((a,b)=>row[b]-row[a]).slice(0,3);const topCols=[...col.keys()].sort((a,b)=>col[b]-col[a]).slice(0,3);const primitives=[];for(const y of topRows)if(row[y]/Math.max(total,1)>.006)primitives.push({type:'horizontal-axis',y:y/h,support:row[y]/Math.max(...row)});for(const x of topCols)if(col[x]/Math.max(total,1)>.006)primitives.push({type:'vertical-axis',x:x/w,support:col[x]/Math.max(...col)});const mask=Uint8Array.from(FG,v=>v>.28);primitives.push(...connectedComponents(mask,w,h,Math.max(12,Math.round(w*h*.0008))));const comps=primitives.filter(p=>p.type==='component');for(const c of comps){const bw=(c.x1-c.x0)*w,bh=(c.y1-c.y0)*h,ratio=bw/(bh+1e-6),cx=c.cx*w,cy=c.cy*h,r=(bw+bh)/4;let ring=0,samples=0;for(let a=0;a<Math.PI*2;a+=Math.PI/36){const x=Math.round(cx+r*Math.cos(a)),y=Math.round(cy+r*Math.sin(a));if(x>=0&&y>=0&&x<w&&y<h){ring+=contours[y*w+x];samples++}}const closure=samples?ring/samples:0;if(ratio>.72&&ratio<1.38&&closure>.16)primitives.push({type:'closed-ellipse',cx:c.cx,cy:c.cy,rx:(c.x1-c.x0)/2,ry:(c.y1-c.y0)/2,support:clamp(closure)})}return primitives}
function drawPrimitives(source,canvas,w,h,primitives,withSource=false){canvas.width=w;canvas.height=h;const ctx=canvas.getContext('2d');ctx.fillStyle=withSource?'#080b0f':'#050608';ctx.fillRect(0,0,w,h);if(withSource){ctx.globalAlpha=.24;ctx.drawImage(source,0,0,w,h);ctx.globalAlpha=1}ctx.lineWidth=Math.max(1,w/320);ctx.strokeStyle='#d6a65f';ctx.fillStyle='#f2d8a6';for(const p of primitives){ctx.globalAlpha=.35+.65*clamp(p.support||.5);ctx.beginPath();if(p.type==='horizontal-axis'){ctx.moveTo(0,p.y*h);ctx.lineTo(w,p.y*h)}else if(p.type==='vertical-axis'){ctx.moveTo(p.x*w,0);ctx.lineTo(p.x*w,h)}else if(p.type==='component'){ctx.rect(p.x0*w,p.y0*h,(p.x1-p.x0)*w,(p.y1-p.y0)*h)}else if(p.type==='closed-ellipse'){ctx.ellipse(p.cx*w,p.cy*h,p.rx*w,p.ry*h,0,0,Math.PI*2)}ctx.stroke()}ctx.globalAlpha=1;for(const p of primitives.filter(p=>p.type==='component')){ctx.beginPath();ctx.arc(p.cx*w,p.cy*h,Math.max(2,w/180),0,Math.PI*2);ctx.fill()}}
function primitiveSummary(primitives){const counts={};for(const p of primitives)counts[p.type]=(counts[p.type]||0)+1;const labels={'vertical-axis':'Verticale assen','horizontal-axis':'Horizontale assen','component':'Dragende componenten','closed-ellipse':'Gesloten ellipsen'};return `<dl>${Object.entries(counts).map(([k,v])=>`<div><dt>${labels[k]||k}</dt><dd>${v}</dd></div>`).join('')}</dl>`}
function draw(a,c,w,h,invert=false){c.width=w;c.height=h;const x=c.getContext('2d'),im=x.createImageData(w,h);for(let i=0;i<a.length;i++){let v=invert?1-a[i]:a[i],p=Math.round(clamp(v)*255),j=i*4;im.data[j]=im.data[j+1]=im.data[j+2]=p;im.data[j+3]=255}x.putImageData(im,0,0)}function copy(a,b){b.width=a.width;b.height=a.height;b.getContext('2d').drawImage(a,0,0)}
function card(parent,title,arr,w,h){let a=document.createElement('article'),h2=document.createElement('h2'),c=document.createElement('canvas');h2.textContent=title;a.append(h2,c);parent.append(a);draw(arr,c,w,h)}
function render(){const{src,w,h,r}=data;copy(src,$('source'));draw(r.contours,$('contours'),w,h);draw(r.FG,$('fg'),w,h);draw(r.LOP,$('lop'),w,h);draw(r.FS,$('fs'),w,h,true);drawPrimitives(src,$('primitiveOverlay'),w,h,r.primitives,true);drawPrimitives(src,$('primitiveCanvas'),w,h,r.primitives,false);$('primitiveSummary').innerHTML=primitiveSummary(r.primitives);$('geometry').innerHTML='';$('light').innerHTML='';for(const op of OPS){card($('geometry'),`G_${op}`,r.G[op],w,h);card($('light'),`L_${op}`,r.LK[op],w,h)}$('results').hidden=false}
async function run(src){$('status').textContent='Geometrische reductie berekenenâ¦';const l=await load(src);data={src:l.c,w:l.w,h:l.h,r:analyze(l.L,l.w,l.h),L:l.L};render();$('status').textContent='Gereed: puur zichtbare geometrie, zonder objectherkenning.'}
// Geintegreerde foto-upload en analyseknop. Ondersteunt zowel de bestaande IDs
// #file / #run als de nieuwere IDs #photoInput / #analyseButton.
const fileControl = $('photoInput') || $('file');
const analyseControl = $('analyseButton') || $('run');

if (fileControl && analyseControl) {
  analyseControl.disabled = true;

  fileControl.onchange = e => {
    selected = e.target.files?.[0] || null;
    analyseControl.disabled = !selected;
    const status = $('status');
    if (status) {
      status.textContent = selected
        ? `Foto gereed voor analyse: ${selected.name}`
        : 'Kies eerst een foto.';
    }
  };

  analyseControl.onclick = async () => {
    if (!selected) {
      const status = $('status');
      if (status) status.textContent = 'Upload eerst een foto.';
      return;
    }

    analyseControl.disabled = true;
    const oldLabel = analyseControl.textContent;
    analyseControl.textContent = 'Analyse bezig...';

    try {
      await run(selected);
      analyseControl.textContent = 'Analyse voltooid';
    } catch (error) {
      console.error(error);
      const status = $('status');
      if (status) status.textContent = `Analyse mislukt: ${error.message}`;
      analyseControl.textContent = 'Probeer opnieuw';
    } finally {
      analyseControl.disabled = false;
      window.setTimeout(() => {
        analyseControl.textContent = oldLabel || 'Analyseer foto';
      }, 1800);
    }
  };
}
$('example').onclick=()=>run('./voorbeeld.jpg');$('exampleLarge').onclick=()=>run('./voorbeeld.jpg');$('detail').oninput=e=>{detail=+e.target.value;if(data){data.r=analyze(data.L,data.w,data.h);render()}};$('threshold').oninput=e=>{threshold=+e.target.value;$('thresholdValue').textContent=threshold.toFixed(2);if(data){data.r=analyze(data.L,data.w,data.h);render()}};document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.querySelectorAll('.view').forEach(x=>x.hidden=true);$(b.dataset.view).hidden=false});if('serviceWorker'in navigator)navigator.serviceWorker.register('./sw.js').catch(()=>{});
