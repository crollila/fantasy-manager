'use strict';
const {app, BrowserWindow, ipcMain, session, Menu, shell, dialog, safeStorage} = require('electron');
const {spawn} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const net = require('node:net');
const crypto = require('node:crypto');
const {teamLink, validOptions} = require('./connection.cjs');
const {EspnSessionStore}=require('./espn-session.cjs');
const {readLeague,allowedEspnUrl,failure}=require('./espn-client.cjs');
let mainWindow, espnWindow, backend, base, token, espnSession, sessionStore, closing=false, quitFlushed=false;
const connectionStates=new Map();
app.setName('Fantasy Manager');
const profileDir=process.env.FANTASY_PROFILE_DIR || (process.env.FANTASY_DATA_DIR ? path.join(process.env.FANTASY_DATA_DIR,'desktop-profile') : path.join(app.getPath('appData'),'Fantasy Manager'));
fs.mkdirSync(profileDir,{recursive:true});
app.setPath('userData',profileDir);
app.setPath('sessionData',profileDir);
const dataDir = process.env.FANTASY_DATA_DIR || path.join(process.env.LOCALAPPDATA || app.getPath('userData'),'Fantasy Manager','data');
const nonce = crypto.randomBytes(24).toString('hex');
app.enableSandbox();
if (!app.requestSingleInstanceLock()) app.quit();
else {
 app.on('second-instance',()=>{ if(mainWindow){mainWindow.restore(); mainWindow.focus();} });
 app.whenReady().then(start).catch(error=>{dialog.showErrorBox('Fantasy Manager could not start', error.message);app.quit();});
}
async function freePort(){return new Promise((resolve,reject)=>{const server=net.createServer();server.on('error',reject);server.listen(0,'127.0.0.1',()=>{const port=server.address().port;server.close(()=>resolve(port));});});}
async function local(route, body){
 const response=await fetch(base+'/api'+route,{method:body===undefined?'GET':'POST',headers:{'Content-Type':'application/json','X-Local-Token':token||''},body:body===undefined?undefined:JSON.stringify(body),signal:AbortSignal.timeout(90000)});
 const result=await response.json();if(!response.ok)throw new Error(typeof result.detail==='string'?result.detail:JSON.stringify(result.detail));return result;
}
function trusted(event){if(!mainWindow||event.sender!==mainWindow.webContents||new URL(event.senderFrame.url).origin!==base)throw new Error('Untrusted application request');}
async function start(){
 fs.mkdirSync(dataDir,{recursive:true});
 espnSession=session.fromPartition('persist:espn');
 // Use the bundled Chromium's web-compatible user agent for the identity provider.
 espnSession.setUserAgent(espnSession.getUserAgent().replace(/\s(?:Electron|Fantasy-Manager|fantasy-manager-desktop)\/[\d.]+/g,''));
 sessionStore=new EspnSessionStore(espnSession,safeStorage,profileDir);
 await sessionStore.initialize();
 const port=await freePort();base=`http://127.0.0.1:${port}`;
 const executable=app.isPackaged?path.join(process.resourcesPath,'backend','FantasyBackend.exe'):path.join(__dirname,'..','.venv','Scripts','python.exe');
 const args=app.isPackaged?[]:[path.join(__dirname,'backend.py')];
 const log=fs.openSync(path.join(dataDir,'desktop.log'),'a');
 backend=spawn(executable,args,{cwd:app.isPackaged?path.dirname(executable):path.join(__dirname,'..'),env:{...process.env,FANTASY_DATA_DIR:dataDir,FANTASY_PORT:String(port),FANTASY_INSTANCE:nonce,PYTHONPATH:app.isPackaged?'':path.join(__dirname,'..')},windowsHide:true,stdio:['ignore',log,log]});
 fs.closeSync(log);
 let spawnError;backend.on('error',e=>{spawnError=e;});
 backend.on('exit',()=>{if(mainWindow&&!closing){dialog.showErrorBox('Local engine stopped','Restart Fantasy Manager. Details are saved in desktop.log in your data folder.');app.quit();}});
 const deadline=Date.now()+90000;let ready=false;
 while(Date.now()<deadline){if(spawnError)throw spawnError;if(backend.exitCode!==null)throw new Error('The local engine exited. See '+path.join(dataDir,'desktop.log'));try{const health=await local('/health');if(health.instance===nonce){ready=true;break;}}catch{}await new Promise(r=>setTimeout(r,300));}
 if(!ready)throw new Error('Local engine startup timed out. See '+path.join(dataDir,'desktop.log'));
 token=(await local('/bootstrap')).token;
 mainWindow=new BrowserWindow({width:1220,height:870,minWidth:820,minHeight:620,title:'Fantasy Manager',icon:path.join(__dirname,'icon.ico'),backgroundColor:'#ffffff',show:false,webPreferences:{preload:path.join(__dirname,'preload.cjs'),contextIsolation:true,nodeIntegration:false,sandbox:true}});
 mainWindow.setMenuBarVisibility(false);
 mainWindow.webContents.setWindowOpenHandler(({url})=>{if(/^https:\/\//.test(url))shell.openExternal(url);return {action:'deny'};});
 mainWindow.webContents.on('will-navigate',(event,url)=>{if(new URL(url).origin!==base)event.preventDefault();});
 ipcMain.handle('connections',event=>{trusted(event);return local('/connections');});
 ipcMain.handle('espn-status',async event=>{trusted(event);return {...await sessionStore.status(),teams:Object.fromEntries(connectionStates)};});
 ipcMain.handle('open-data',event=>{trusted(event);return shell.openPath(dataDir);});
 ipcMain.handle('espn-refresh',async(event,options)=>{trusted(event);const {team,week}=validOptions(options);if(!team)throw new Error('Choose a connected team.');return sync(team,week);});
 ipcMain.handle('espn-connect',async(event,options)=>{trusted(event);return connect(options);});
 ipcMain.handle('refresh-saved',async event=>{
  trusted(event);const report=await local('/intelligence'),connections=await local('/connections');let synced=0;const errors=[];
  const week=report.week||1;
  for(const connection of connections){
   const target=teamLink(connection.url);if(!target||(report.season&&target.season!==report.season))continue;
   try {await sync(target,week);synced++;try{await local(`/leagues/${target.league_id}/lineup`,{week,risk:'balanced',refresh_injuries:false});}catch{errors.push({league_id:target.league_id,code:'lineup_unavailable',message:connection.name+': roster saved; lineup data are not ready yet.'});}}
   catch(error){errors.push({league_id:target.league_id,code:error.code||'sync_error',message:connection.name+': '+error.message});}
  }
  return {synced,errors};
 });
 await mainWindow.loadURL(base);mainWindow.show();
 if(process.env.FANTASY_SMOKE_TEST){await mainWindow.webContents.executeJavaScript('document.title');fs.writeFileSync(path.join(dataDir,'desktop-smoke.json'),JSON.stringify({ok:true,version:app.getVersion(),title:await mainWindow.webContents.executeJavaScript('document.title'),url:base}));app.quit();}
}
function notifyConnection(team,state){
 if(team)connectionStates.set(team.league_id,state);
 if(mainWindow&&!mainWindow.isDestroyed())mainWindow.webContents.send('espn-connection-event',{league_id:team?.league_id,...state});
}
async function sync(team,week){
 try {
  const snapshot=await readLeague((...args)=>espnSession.fetch(...args),team,week);
  let result;
  try{result=await local('/espn/import',{...team,week,snapshot});}
  catch(error){throw failure('roster_error','ESPN answered, but the league could not be imported: '+error.message);}
  await sessionStore.flush();
  notifyConnection(team,{state:'connected',message:'Team synced. ESPN sign-in is remembered on this computer.',last_success:new Date().toISOString()});
  return result;
 } catch(error){notifyConnection(team,{state:error.code==='sign_in_required'?'sign_in_required':'retry',code:error.code||'sync_error',message:error.message});throw error;}
}
function secureEspnWindow(win,onReturn){
 const navigation=(event,url)=>{if(!allowedEspnUrl(url))event.preventDefault();};
 win.webContents.on('will-navigate',navigation);
 win.webContents.on('will-redirect',navigation);
 win.webContents.setWindowOpenHandler(({url})=>{
  if(url!=='about:blank'&&!allowedEspnUrl(url))return {action:'deny'};
  return {action:'allow',overrideBrowserWindowOptions:{parent:win,width:650,height:800,webPreferences:{session:espnSession,nodeIntegration:false,contextIsolation:true,sandbox:true,preload:undefined}}};
 });
 win.webContents.on('did-create-window',child=>{
  secureEspnWindow(child,onReturn);
  child.on('closed',()=>{void sessionStore.flush();onReturn();});
 });
}
async function connect(options){
 const {team,week}=validOptions(options);
 if(espnWindow&&!espnWindow.isDestroyed()){espnWindow.focus();throw new Error('The ESPN window is already open. Finish connecting there.');}
 if(team){
  try{return await sync(team,week);}catch(error){if(!['sign_in_required','access_denied','unexpected_response'].includes(error.code))throw error;}
 }
 return new Promise(resolve=>{
  let result={cancelled:true},importing=false,lastAttempt=0,finished=false;
  const win=espnWindow=new BrowserWindow({width:1200,height:850,title:'ESPN — open My Team to connect',webPreferences:{session:espnSession,nodeIntegration:false,contextIsolation:true,sandbox:true}});
  async function importTeam(manual=false){
   if(importing||win.isDestroyed()||finished||(!manual&&Date.now()-lastAttempt<5000))return;
   const target=teamLink(win.webContents.getURL())||team;
   if(!target){if(manual)await dialog.showMessageBox(win,{message:'Open your ESPN My Team page, then choose Retry import.'});return;}
   importing=true;lastAttempt=Date.now();
   try{result=await sync(target,week);finished=true;clearInterval(poll);await sessionStore.flush();win.close();}
   catch(error){if(!win.isDestroyed()){win.setTitle('ESPN — '+error.message);if(manual)await dialog.showMessageBox(win,{message:error.message});}}
   finally{importing=false;}
  }
  secureEspnWindow(win,()=>importTeam(false));
  win.setMenu(Menu.buildFromTemplate([{label:'Retry import',click:()=>importTeam(true)},{label:'My Team',enabled:!!team,click:()=>win.loadURL(options.url)},{label:'Reload ESPN',click:()=>win.reload()},{label:'Close',click:()=>win.close()}]));
  win.webContents.on('did-finish-load',()=>importTeam(false));
  win.webContents.on('did-navigate-in-page',()=>importTeam(false));
  win.webContents.on('did-fail-load',(_event,code,_description,_url,isMainFrame)=>{if(isMainFrame&&code!==-3&&!win.isDestroyed())win.setTitle('ESPN could not load — check your connection, then Reload ESPN');});
  const poll=setInterval(()=>importTeam(false),5000);
  const cookieChanged=(_event,cookie)=>{if(cookie.name==='espn_s2')void importTeam(false);};
  espnSession.cookies.on('changed',cookieChanged);
  win.on('closed',()=>{clearInterval(poll);espnSession.cookies.removeListener('changed',cookieChanged);espnWindow=null;void sessionStore.flush().finally(()=>resolve(result));});
  win.loadURL(team?options.url:'https://www.espn.com/fantasy/football/').catch(()=>{});
 });
}
app.on('before-quit',event=>{
 if(quitFlushed){closing=true;if(backend&&!backend.killed)backend.kill();return;}
 event.preventDefault();if(closing)return;closing=true;
 Promise.race([sessionStore?.flush().catch(()=>{}),new Promise(resolve=>setTimeout(resolve,3000))]).finally(()=>{quitFlushed=true;app.quit();});
});
app.on('window-all-closed',()=>app.quit());
