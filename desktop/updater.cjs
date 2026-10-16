'use strict';
// Automatic updates from GitHub Releases (the `publish` block in package.json). A new release is
// downloaded in the background and installed when the user presses the header's update button
// or simply quits the app, so nobody has to uninstall and run the installer again.
const fs=require('node:fs');
const CHECK_EVERY=6*60*60*1000;

function fileLogger(file){
 const write=(level,args)=>{try{fs.appendFileSync(file,`${new Date().toISOString()} ${level} ${args.map(a=>a instanceof Error?a.message:typeof a==='string'?a:JSON.stringify(a)).join(' ')}\n`);}catch{}};
 return {info:(...a)=>write('info',a),warn:(...a)=>write('warn',a),error:(...a)=>write('error',a),debug:()=>{}};
}

function createUpdater({autoUpdater,currentVersion,send,logFile,beforeInstall,timers={setTimeout,setInterval}}){
 let status={state:'idle',current:currentVersion},installWhenReady=false;
 const set=next=>{status={...next,current:currentVersion};send(status);};
 if(logFile)autoUpdater.logger=fileLogger(logFile);
 autoUpdater.autoDownload=true;
 autoUpdater.autoInstallOnAppQuit=true;
 autoUpdater.on('checking-for-update',()=>{if(status.state!=='ready')set({state:'checking'});});
 autoUpdater.on('update-available',info=>set({state:'downloading',version:info.version,percent:0}));
 autoUpdater.on('update-not-available',()=>set({state:'current'}));
 autoUpdater.on('download-progress',p=>{if(status.state==='downloading')set({...status,percent:Math.round(p.percent||0)});});
 autoUpdater.on('update-downloaded',info=>{set({state:'ready',version:info.version});if(installWhenReady)void api.install().catch(()=>{});});
 // Offline, GitHub unavailable or rate-limited: keep running and try again on the next check.
 autoUpdater.on('error',error=>{if(status.state!=='ready')set({state:'error',error:String(error?.message||error)});});
 const check=()=>{if(status.state==='ready'||status.state==='downloading')return;autoUpdater.checkForUpdates().catch(()=>{});};
 const api={
  status:()=>status,
  check,
  // Checked as soon as the window opens, then every six hours while the app stays open.
  schedule(delay=0){timers.setTimeout(check,delay);timers.setInterval(check,CHECK_EVERY);},
  async install(){
   // Pressed while the download is still running: install as soon as it finishes.
   if(status.state==='downloading'){installWhenReady=true;set({...status,installing:true});return;}
   if(status.state!=='ready')throw new Error('No downloaded update to install');
   set({...status,installing:true});
   // The bundled engine holds files in the install folder; stop it so the installer can replace them.
   await beforeInstall();
   autoUpdater.quitAndInstall(true,true);
  }
 };
 return api;
}

module.exports={createUpdater,fileLogger,CHECK_EVERY};
