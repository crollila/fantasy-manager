chrome.runtime.onMessage.addListener((message,sender,respond)=>{
 (async()=>{
  if(message.type==='sync-league'){
   if(sender.id!==chrome.runtime.id||sender.tab)throw new Error('League sync must originate in the extension popup');
   const {config}=await chrome.storage.local.get('config');
   if(!config?.token||!/^\d+$/.test(config.leagueId)||!Number.isInteger(config.myTeamId))throw new Error('Save pairing token, league ID and your ESPN team ID first');
   const season=Number(config.season||2026),week=Number(config.week||1);
   if(!Number.isInteger(season)||season<2005||season>2100||!Number.isInteger(week)||week<1||week>18)throw new Error('Invalid season/week');
   const url=new URL(`https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/${season}/segments/0/leagues/${config.leagueId}`);
   for(const view of ['mSettings','mTeam','mRoster','mMatchup','mStatus'])url.searchParams.append('view',view);
   url.searchParams.set('scoringPeriodId',week);
   const remote=await fetch(url,{credentials:'include',signal:AbortSignal.timeout(20000)});
   if(!remote.ok)throw new Error(`ESPN access failed (${remote.status}). Log into ESPN in this Chrome profile and verify league membership.`);
   const snapshot=await remote.json();
   const local=await fetch('http://127.0.0.1:8000/api/espn/import',{method:'POST',headers:{'Content-Type':'application/json','X-Local-Token':config.token},body:JSON.stringify({league_id:config.leagueId,my_team_id:config.myTeamId,season,week,snapshot}),signal:AbortSignal.timeout(20000)});
   const result=await local.json();if(!local.ok)throw new Error(typeof result.detail==='string'?result.detail:JSON.stringify(result.detail));
   const status={ok:true,message:`Synced ${result.league.name}: ${result.players_synced} rostered players, week ${week}`,at:new Date().toISOString()};
   await chrome.storage.local.set({status});respond(status);return;
  }
  if(!sender.tab?.url?.startsWith('https://fantasy.espn.com/football/'))throw new Error('Unexpected message sender');
  if(message.type==='diagnostic'){await chrome.storage.local.set({status:{ok:false,message:message.message,at:new Date().toISOString()}});respond({ok:true});return;}
  if(message.type!=='snapshot')return;
  const {config}=await chrome.storage.local.get('config');
  if(!config?.enabled||!config.token||!config.leagueId)throw new Error('Pair the local app first');
  const actualLeague=new URL(sender.tab.url).searchParams.get('leagueId');
  if(actualLeague!==config.leagueId)throw new Error('ESPN league mismatch');
  const response=await fetch(`http://127.0.0.1:8000/api/leagues/${encodeURIComponent(config.leagueId)}/espn`,{method:'POST',headers:{'Content-Type':'application/json','X-Local-Token':config.token},body:JSON.stringify(message.payload),signal:AbortSignal.timeout(7000)});
  const data=await response.json();
  if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));
  await chrome.storage.local.set({status:{ok:true,message:`Synced ${data.count} picks · revision ${data.revision}`,at:new Date().toISOString()}});
  await chrome.action.setBadgeText({text:'OK'});await chrome.action.setBadgeBackgroundColor({color:'#28765d'});respond({ok:true});
 })().catch(async e=>{await chrome.storage.local.set({status:{ok:false,message:e.message,at:new Date().toISOString()}});await chrome.action.setBadgeText({text:'!'});await chrome.action.setBadgeBackgroundColor({color:'#a54343'});respond({ok:false,error:e.message});});
 return true;
});
