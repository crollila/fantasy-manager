const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const os=require('node:os');
const path=require('node:path');
const crypto=require('node:crypto');
const {EventEmitter}=require('node:events');
const {EspnSessionStore,authCookie,cookieDetails}=require('./espn-session.cjs');
const {readLeague,allowedEspnUrl}=require('./espn-client.cjs');

function fixtureSession(initial=[]){
 const cookies=new EventEmitter();cookies.rows=initial;cookies.get=async()=>cookies.rows;
 cookies.set=async c=>{cookies.rows.push({...c,domain:c.domain||new URL(c.url).hostname,hostOnly:!c.domain});};
 cookies.flushStore=async()=>{};return {cookies,flushStorageData(){}};
}
function encryptor(){
 const key=crypto.randomBytes(32);
 return {isEncryptionAvailable:()=>true,encryptString:plain=>{const iv=crypto.randomBytes(12),cipher=crypto.createCipheriv('aes-256-gcm',key,iv);return Buffer.concat([iv,cipher.update(plain),cipher.final(),cipher.getAuthTag()]);},decryptString:data=>{const decipher=crypto.createDecipheriv('aes-256-gcm',key,data.subarray(0,12));decipher.setAuthTag(data.subarray(-16));return Buffer.concat([decipher.update(data.subarray(12,-16)),decipher.final()]).toString();}};
}
const login={name:'espn_s2',value:'SYNTHETIC-SESSION',domain:'.espn.com',path:'/',secure:true,httpOnly:true,sameSite:'no_restriction',session:true};

test('remembered ESPN session restores across fresh browser sessions without plaintext',async()=>{
 const dir=await fs.mkdtemp(path.join(os.tmpdir(),'fm-session-')),safe=encryptor();
 try{
  const first=new EspnSessionStore(fixtureSession([login,{...login,name:'tracking_id'}]),safe,dir);
  await first.initialize();await first.flush();
  const encrypted=await fs.readFile(path.join(dir,'espn-session.enc'));
  assert.equal(encrypted.includes(Buffer.from(login.value)),false);
  const secondSession=fixtureSession(),second=new EspnSessionStore(secondSession,safe,dir);await second.initialize();
  assert.equal(secondSession.cookies.rows.length,1);assert.equal((await second.status()).hasSession,true);
  assert.equal(secondSession.cookies.rows[0].httpOnly,true);
 }finally{await fs.rm(dir,{recursive:true,force:true});}
});
test('logout is not resurrected and expired cookies are not restored',async()=>{
 const dir=await fs.mkdtemp(path.join(os.tmpdir(),'fm-session-')),safe=encryptor(),session=fixtureSession([login]);
 try{
  const store=new EspnSessionStore(session,safe,dir);await store.initialize();await store.flush();
  session.cookies.rows=[];session.cookies.emit('changed',{},login,'explicit',true);await store.pending;
  assert.equal((await new EspnSessionStore(fixtureSession(),safe,dir).initialize()),undefined);
  await assert.rejects(fs.access(path.join(dir,'espn-session.enc')));
  session.cookies.rows=[{...login,expirationDate:1}];await store.flush();await assert.rejects(fs.access(path.join(dir,'espn-session.enc')));
 }finally{await fs.rm(dir,{recursive:true,force:true});}
});
test('existing fresh browser credentials are never overwritten by older vault contents',async()=>{
 const dir=await fs.mkdtemp(path.join(os.tmpdir(),'fm-session-')),safe=encryptor();
 try{
  await new EspnSessionStore(fixtureSession([login]),safe,dir).flush();
  const session=fixtureSession([{...login,value:'NEW-SYNTHETIC'}]);await new EspnSessionStore(session,safe,dir).initialize();
  assert.equal(session.cookies.rows.length,1);assert.equal(session.cookies.rows[0].value,'NEW-SYNTHETIC');
 }finally{await fs.rm(dir,{recursive:true,force:true});}
});
test('secrets have no plaintext fallback and host-only cookies stay host-only',async()=>{
 assert.equal(authCookie({...login,domain:'.espn.com.evil.test'}),false);
 assert.equal(cookieDetails({...login,hostOnly:true}).domain,undefined);
 const dir=await fs.mkdtemp(path.join(os.tmpdir(),'fm-session-'));
 try{const store=new EspnSessionStore(fixtureSession([login]),{isEncryptionAvailable:()=>false},dir);await store.initialize();await store.flush();assert.equal((await store.status()).remembered,false);await assert.rejects(fs.access(path.join(dir,'espn-session.enc')));}
 finally{await fs.rm(dir,{recursive:true,force:true});}
});
test('network, rate limits and roster errors are not mistaken for expired login',async()=>{
 const team={league_id:'123',season:2026};
 for(const [status,code] of [[401,'sign_in_required'],[403,'access_denied'],[429,'rate_limited'],[500,'provider_unavailable'],[404,'league_unavailable']]){
  await assert.rejects(readLeague(async()=>({ok:false,status}),team,1),e=>e.code===code);
 }
 await assert.rejects(readLeague(async()=>{throw Error('internal transport info');},team,1),e=>e.code==='network'&&!e.message.includes('internal'));
 await assert.rejects(readLeague(async()=>({ok:true,json:async()=>({})}),team,1),e=>e.code==='invalid_snapshot');
});
test('Disney authentication popups allow identity hosts, rejecting lookalikes',()=>{
 for(const url of ['https://registerdisney.go.com/login','https://mydisney.com/login','https://fantasy.espn.com/football/team'])assert.equal(allowedEspnUrl(url),true);
 for(const url of ['https://espn.com.evil.test','http://espn.com','file:///secret','javascript:alert(1)'])assert.equal(allowedEspnUrl(url),false);
});
