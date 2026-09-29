const {chromium}=require('/opt/node22/lib/node_modules/playwright');const fs=require('fs');
(async()=>{const b=await chromium.launch({executablePath:'/opt/pw-browsers/chromium'});
const p=await b.newPage({viewport:{width:1920,height:1080}});
await p.goto('file://'+__dirname+'/promo.html');await p.waitForTimeout(800);
fs.mkdirSync('f',{recursive:true});const fps=30,N=20*fps;
for(let i=0;i<N;i++){await p.evaluate(t=>seek(t),i/fps);await p.screenshot({path:`f/${String(i).padStart(4,'0')}.jpg`,type:'jpeg',quality:92});}
await b.close()})();
