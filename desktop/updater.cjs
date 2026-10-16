'use strict';
// Automatic updates from GitHub Releases (the `publish` block in package.json). A new release is
// downloaded in the background and installed when the user restarts from the banner or simply
// quits the app, so nobody has to uninstall and run the installer again.
const fs=require('node:fs');
const CHECK_EVERY=6*60*60*1000;

function fileLogger(file){
 const write=(level,args)=>{try{fs.appendFileSync(file,`${new Date().toISOString()} ${level} ${args.map(a=>a instanceof Error?a.message:typeof a==='string'?a:JSON.stringify(a)).join(' ')}\n`);}catch{}};
 return {info:(...a)=>write('info',a),warn:(...a)=>write('warn',a),error:(...a)=>write('error',a),debug:()=>{}};
}

function createUpdater({autoUpdater,currentVersion,send,logFile,beforeInstall,timers={setTimeout,setInterval}}){
 let status={state:'idle',current:currentVersion};
 const set=next=>{status={...next,current:currentVersion};send(status);};
 if(logFile)autoUpdater.logger=fileLogger(logFile);
 autoUpdater.autoDownload=true;
 autoUpdater.autoInstallOnAppQuit=true;
 autoUpdater.on('checking-for-update',()=>{if(status.state!=='ready')set({state:'checking'});});
 autoUpdater.on('update-available',info=>set({state:'downloading',version:info.version,percent:0}));
 autoUpdater.on('update-not-available',()=>set({state:'current'}));
 autoUpdater.on('download-progress',p=>{if(status.state==='downloading')set({...status,percent:Math.round(p.percent||0)});});
 autoUpdater.on('update-downloaded',info=>set({state:'ready',version:info.version}));
 // Offline, GitHub unavailable or rate-limited: keep running and try again on the next check.
 autoUpdater.on('error',error=>{if(status.state!=='ready')set({state:'error',error:String(error?.message||error)});});
 const check=()=>{if(status.state==='ready'||status.state==='downloading')return;autoUpdater.checkForUpdates().catch(()=>{});};
 return {
  status:()=>status,
  check,
  schedule(delay=10000){timers.setTimeout(check,delay);timers.setInterval(check,CHECK_EVERY);},
  async install(){
   if(status.state!=='ready')throw new Error('No downloaded update to install');
   // The bundled engine holds files in the install folder; stop it so the installer can replace them.
   await beforeInstall();
   autoUpdater.quitAndInstall(true,true);
  }
 };
}

module.exports={createUpdater,fileLogger,CHECK_EVERY};
