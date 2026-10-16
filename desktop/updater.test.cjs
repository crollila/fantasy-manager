const {test}=require('node:test');const assert=require('node:assert/strict');const {EventEmitter}=require('node:events');
const {createUpdater}=require('./updater.cjs');
function fake(){const u=new EventEmitter();u.checks=0;u.installed=null;u.checkForUpdates=async()=>{u.checks++;};u.quitAndInstall=(silent,runAfter)=>{u.installed={silent,runAfter};};return u;}
test('downloads in the background, installs on quit and reports each stage',()=>{
 const u=fake(),sent=[];const updater=createUpdater({autoUpdater:u,currentVersion:'0.7.1',send:s=>sent.push(s),beforeInstall:async()=>{}});
 assert.equal(u.autoDownload,true);assert.equal(u.autoInstallOnAppQuit,true);
 u.emit('checking-for-update');u.emit('update-available',{version:'0.7.2'});u.emit('download-progress',{percent:41.6});u.emit('update-downloaded',{version:'0.7.2'});
 assert.deepEqual(sent.map(s=>s.state),['checking','downloading','downloading','ready']);
 assert.equal(sent[2].percent,42);assert.deepEqual(updater.status(),{state:'ready',version:'0.7.2',current:'0.7.1'});
 // A later check or a transient error must not hide the downloaded update.
 updater.check();u.emit('error',new Error('offline'));assert.equal(u.checks,0);assert.equal(updater.status().state,'ready');
});
test('restart stops the engine before running the installer silently and relaunching',async()=>{
 const u=fake(),order=[];const updater=createUpdater({autoUpdater:u,currentVersion:'0.7.1',send:()=>{},beforeInstall:async()=>{order.push('engine stopped');}});
 await assert.rejects(updater.install());assert.equal(u.installed,null);
 u.emit('update-downloaded',{version:'0.7.2'});
 const quit=u.quitAndInstall;u.quitAndInstall=(...a)=>{order.push('install');quit(...a);};
 await updater.install();assert.deepEqual(order,['engine stopped','install']);assert.deepEqual(u.installed,{silent:true,runAfter:true});
});
test('offline checks are reported and retried on schedule',()=>{
 const u=fake(),timers={timeouts:[],intervals:[],setTimeout:(f,d)=>timers.timeouts.push([f,d]),setInterval:(f,d)=>timers.intervals.push([f,d])};
 const updater=createUpdater({autoUpdater:u,currentVersion:'0.7.1',send:()=>{},beforeInstall:async()=>{},timers});
 updater.schedule(5);timers.timeouts[0][0]();u.emit('error',new Error('net::ERR_INTERNET_DISCONNECTED'));
 assert.equal(updater.status().state,'error');timers.intervals[0][0]();assert.equal(u.checks,2);assert.equal(timers.intervals[0][1],6*60*60*1000);
});
test('pressing update during the download installs once it finishes',async()=>{
 const u=fake(),sent=[];const updater=createUpdater({autoUpdater:u,currentVersion:'0.7.1',send:s=>sent.push(s),beforeInstall:async()=>{}});
 u.emit('update-available',{version:'0.7.2'});await updater.install();
 assert.equal(u.installed,null);assert.equal(updater.status().installing,true);
 u.emit('update-downloaded',{version:'0.7.2'});await new Promise(r=>setImmediate(r));
 assert.deepEqual(u.installed,{silent:true,runAfter:true});
});
test('checks immediately at launch',()=>{
 const u=fake(),timers={timeouts:[],setTimeout:(f,d)=>timers.timeouts.push([f,d]),setInterval:()=>{}};
 createUpdater({autoUpdater:u,currentVersion:'0.7.1',send:()=>{},beforeInstall:async()=>{},timers}).schedule();
 assert.equal(timers.timeouts[0][1],0);
});
