'use strict';
const {app, BrowserWindow, ipcMain, session, Menu, shell, dialog} = require('electron');
const {spawn} = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const net = require('node:net');
const crypto = require('node:crypto');
const {teamLink, validOptions} = require('./connection.cjs');
let mainWindow, espnWindow, backend, base, token, closing=false;
app.setName('Fantasy Manager');
app.setPath('userData',path.join(app.getPath('appData'),'Fantasy Manager'));
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
 mainWindow=new BrowserWindow({width:1220,height:870,minWidth:820,minHeight:620,title:'Fantasy Manager',icon:path.join(__dirname,'icon.ico'),backgroundColor:'#f5f7fa',show:false,webPreferences:{preload:path.join(__dirname,'preload.cjs'),contextIsolation:true,nodeIntegration:false,sandbox:true}});
 mainWindow.setMenuBarVisibility(false);
 mainWindow.webContents.setWindowOpenHandler(({url})=>{if(/^https:\/\//.test(url))shell.openExternal(url);return {action:'deny'};});
 mainWindow.webContents.on('will-navigate',(event,url)=>{if(new URL(url).origin!==base)event.preventDefault();});
 ipcMain.handle('connections',event=>{trusted(event);return local('/connections');});
 ipcMain.handle('open-data',event=>{trusted(event);return shell.openPath(dataDir);});
 ipcMain.handle('espn-refresh',async(event,options)=>{trusted(event);const {team,week}=validOptions(options);if(!team)throw new Error('Choose a connected team.');return sync(team,week);});
 ipcMain.handle('espn-connect',async(event,options)=>{trusted(event);return connect(options);});
 ipcMain.handle('refresh-saved',async event=>{
  trusted(event);const report=await local('/intelligence'),connections=await local('/connections');let synced=0;const errors=[];
  for(const connection of connections){const target=teamLink(connection.url);if(!target||target.season!==report.season)continue;try{await sync(target,report.week);await local(`/leagues/${target.league_id}/lineup`,{week:report.week,risk:'balanced',refresh_injuries:false});synced++;}catch{errors.push(connection.name+' needs sign-in or roster data');}}
  return {synced,errors};
 });
 await mainWindow.loadURL(base);mainWindow.show();
 if(process.env.FANTASY_SMOKE_TEST){await mainWindow.webContents.executeJavaScript('document.title');fs.writeFileSync(path.join(dataDir,'desktop-smoke.json'),JSON.stringify({ok:true,version:app.getVersion(),title:await mainWindow.webContents.executeJavaScript('document.title'),url:base}));app.quit();}
}
async function sync(team,week){
 const remote=new URL(`https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/${team.season}/segments/0/leagues/${team.league_id}`);
 for(const view of ['mSettings','mTeam','mRoster','mMatchup','mStatus'])remote.searchParams.append('view',view);
 remote.searchParams.set('scoringPeriodId',String(week));
 const response=await session.fromPartition('persist:espn').fetch(remote.href,{credentials:'include',signal:AbortSignal.timeout(30000)});
 if(!response.ok)throw new Error('ESPN needs you to sign in. Choose Connect ESPN, sign in, and open My Team.');
 const snapshot=await response.json();
 return local('/espn/import',{...team,week,snapshot});
}
async function connect(options){
 const {team,week}=validOptions(options);
 if(espnWindow&&!espnWindow.isDestroyed()){espnWindow.focus();throw new Error('An ESPN connection window is already open. Finish that connection first.');}
 return new Promise(resolve=>{
  let result={cancelled:true},importing=false;
  const win=espnWindow=new BrowserWindow({width:1200,height:850,title:'ESPN — sign in, open My Team, then choose Import this team',webPreferences:{partition:'persist:espn',nodeIntegration:false,contextIsolation:true,sandbox:true}});
  const allowed=url=>{try{const u=new URL(url);return u.protocol==='https:'&&(u.hostname==='espn.com'||u.hostname.endsWith('.espn.com')||u.hostname==='go.com'||u.hostname.endsWith('.go.com')||u.hostname==='disney.com'||u.hostname.endsWith('.disney.com'));}catch{return false;}};
  win.webContents.on('will-navigate',(event,url)=>{if(!allowed(url))event.preventDefault();});
  win.webContents.setWindowOpenHandler(({url})=>{if(allowed(url))win.loadURL(url);return {action:'deny'};});
  async function importTeam(manual=false){
   if(importing||win.isDestroyed())return;
   const target=teamLink(win.webContents.getURL());
   if(!target){if(manual)dialog.showMessageBox(win,{message:'Open your ESPN My Team page, then choose Import this team.'});return;}
   importing=true;
   try{result=await sync(target,week);win.close();}catch(error){if(manual&&!win.isDestroyed())dialog.showMessageBox(win,{message:error.message});}finally{importing=false;}
  }
  win.setMenu(Menu.buildFromTemplate([{label:'Import this team',click:()=>importTeam(true)},{label:'Reload ESPN',click:()=>win.reload()},{label:'Close',click:()=>win.close()}]));
  win.webContents.on('did-finish-load',()=>importTeam(false));
  win.webContents.on('did-navigate-in-page',()=>importTeam(false));
  win.on('closed',()=>{espnWindow=null;resolve(result);});
  win.loadURL(team?options.url:'https://www.espn.com/fantasy/football/').catch(()=>{});
 });
}
app.on('before-quit',()=>{closing=true;if(backend&&!backend.killed)backend.kill();});
app.on('window-all-closed',()=>app.quit());
