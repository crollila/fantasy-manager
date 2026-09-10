import {_electron as electron} from '@playwright/test';
import path from 'node:path';
import fs from 'node:fs/promises';
const root=path.resolve('..');
const dataDir=path.join(root,'storage','espn-restart-'+Date.now());
const teamUrl='https://fantasy.espn.com/football/team?leagueId=123&teamId=5&seasonId=2026';
const snapshot={id:123,seasonId:2026,settings:{name:'Restart test league',rosterSettings:{lineupSlotCounts:{4:1,20:1}},scoringSettings:{scoringItems:[{statId:42,points:.1},{statId:53,points:1}]}},teams:[{id:5,name:'Restart test team',roster:{entries:[{lineupSlotId:4,playerPoolEntry:{player:{id:101,fullName:'Test Receiver',defaultPositionId:3,injuryStatus:'ACTIVE',stats:[{seasonId:2026,scoringPeriodId:1,statSourceId:1,appliedTotal:12}]}}}]}},{id:9,name:'Other team',roster:{entries:[]}}]};
async function launch(){
 const packaged=process.env.FANTASY_TEST_EXE;
 const instance=await electron.launch({executablePath:packaged||path.join(root,'desktop','node_modules','electron','dist','electron.exe'),args:packaged?[]:[path.join(root,'desktop')],env:{...process.env,FANTASY_DISABLE_AUTO_REFRESH:'1',FANTASY_DATA_DIR:dataDir},timeout:90000});
 const page=await instance.firstWindow({timeout:90000});await page.getByRole('heading',{name:'Your weekly lineup',exact:true}).waitFor();
 await instance.evaluate(async({session},{snapshot,teamUrl})=>{
  const espn=session.fromPartition('persist:espn');
  await espn.protocol.handle('https',async request=>{
   const url=new URL(request.url);
   if(url.hostname==='lm-api-reads.fantasy.espn.com'){
    const authenticated=(await espn.cookies.get({name:'espn_s2'})).some(c=>c.value==='SYNTHETIC-RESTART-SESSION');
    return new Response(JSON.stringify(authenticated?snapshot:{error:'login required'}),{status:authenticated?200:401,headers:{'content-type':'application/json'}});
   }
   if(url.hostname==='registerdisney.go.com')return new Response('<button id="finish" onclick="window.opener.postMessage(\'fixture-login-complete\',\'https://www.espn.com\');window.close()">Finish fixture login</button>',{headers:{'content-type':'text/html'}});
   if(url.hostname.endsWith('espn.com'))return new Response(`<button id="login" onclick="window.open('https://registerdisney.go.com/fixture','fixture-login')">Sign in fixture</button><script>addEventListener('message',e=>{if(e.origin==='https://registerdisney.go.com'&&e.data==='fixture-login-complete')location.href=${JSON.stringify(teamUrl)}})</script>`,{headers:{'content-type':'text/html'}});
   return new Response('No external network in fixture',{status:404});
  });
 },{snapshot,teamUrl});
 return {instance,page};
}
let first=await launch();
try{
 await first.page.getByRole('button',{name:'Manage ESPN',exact:true}).click();
 await first.page.getByRole('button',{name:'Open ESPN',exact:true}).click();
 await first.instance.waitForEvent('window',{predicate:p=>p.url().startsWith('https://www.espn.com'),timeout:15000}).catch(()=>{});
 const espn=first.instance.windows().find(p=>p.url().startsWith('https://www.espn.com'));
 if(!espn)throw Error('ESPN connection window did not open');
 await espn.getByRole('button',{name:'Sign in fixture'}).click();
 const popup=await first.instance.waitForEvent('window',{predicate:p=>p.url().startsWith('https://registerdisney.go.com'),timeout:5000}).catch(()=>first.instance.windows().find(p=>p.url().startsWith('https://registerdisney.go.com')));
 await popup.getByRole('button',{name:'Finish fixture login'}).waitFor();
 if(!await popup.evaluate(()=>!!window.opener))throw Error('Login popup lost its opener');
 await first.instance.evaluate(async({session})=>{
  const espn=session.fromPartition('persist:espn');
  await espn.cookies.set({url:'https://fantasy.espn.com',domain:'.espn.com',name:'espn_s2',value:'SYNTHETIC-RESTART-SESSION',path:'/',secure:true,httpOnly:true,sameSite:'no_restriction'});
  await espn.cookies.set({url:'https://fantasy.espn.com',domain:'.espn.com',name:'SWID',value:'SYNTHETIC-SWID',path:'/',secure:true});
 });
 await popup.getByRole('button',{name:'Finish fixture login'}).click();
 await first.page.getByRole('button',{name:/UP TO DATE Restart test team/}).waitFor({timeout:20000});
 await first.page.screenshot({path:path.join(root,'storage','espn-connected-professional.png'),fullPage:true});
}finally{await first.instance.close();}
const encrypted=await fs.readFile(path.join(dataDir,'desktop-profile','espn-session.enc'));
if(encrypted.includes(Buffer.from('SYNTHETIC-RESTART-SESSION')))throw Error('Session written in plaintext');
const second=await launch();
try{
 await second.page.getByRole('button',{name:/SAVED TEAM Restart test team/}).waitFor();
 const status=await second.page.evaluate(()=>window.fantasyDesktop.status());
 if(!status.hasSession)throw Error('Session was not restored after full restart');
 const result=await second.page.evaluate(teamUrl=>window.fantasyDesktop.refresh({url:teamUrl,week:1}),teamUrl);
 if(result.league?.id!=='123')throw Error('Remembered session did not authorize league import');
 if(second.instance.windows().length!==1)throw Error('Unnecessary login window on reconnect');
 await second.page.getByRole('button',{name:/UP TO DATE Restart test team/}).waitFor();
 await second.page.screenshot({path:path.join(root,'storage','espn-reopened-professional.png'),fullPage:true});
 console.log('ESPN popup, delayed login import, encrypted session, saved team, full restart and silent reconnect passed.');
}finally{await second.instance.close();}
