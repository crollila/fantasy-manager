/* Pure, fail-closed DOM parser. Selectors can be adapted without shipping credentials. */
(function(root){
 function parseDraft(doc, config={}){
  const selector=config.rowSelector||'[data-pick-number], .draft-pick, .DraftPick, [data-testid="draft-pick"]';
  const rows=Array.from(doc.querySelectorAll(selector));
  const errors=[],picks=[],seen=new Map();
  for(const row of rows){
   const text=(row.textContent||'').replace(/\s+/g,' ').trim();
   const pickText=row.getAttribute('data-pick-number')||row.querySelector(config.pickSelector||'.pick-number, [data-testid="pick-number"]')?.textContent||text.match(/(?:Overall\s+Pick|Pick)\s*#?\s*(\d+)/i)?.[1];
   const number=Number(pickText?.match(/\d+/)?.[0]);
   const playerLink=row.querySelector('a[href*="/id/"], a[href*="playerId="]');
   const href=playerLink?.getAttribute('href')||'';
   const espn_id=row.getAttribute('data-player-id')||href.match(/(?:\/id\/|playerId=)(\d+)/)?.[1];
   if(!number&&!espn_id)continue; // Empty, future draft cells are allowed.
   const teamText=(row.getAttribute('data-team-name')||row.querySelector(config.teamSelector||'.team-name, [data-testid="team-name"]')?.textContent||'').trim();
   const teamKey=row.getAttribute('data-team-id');
   const team=config.teamMap?.[teamKey]??config.teamMap?.[teamText];
   if(!Number.isInteger(number)||number<1||!espn_id||!/^\d+$/.test(espn_id)||!Number.isInteger(team)){
    errors.push(`Unresolved draft row: ${text.slice(0,120)}. Need overall pick, ESPN player ID and explicit team mapping.`);continue;
   }
   const item={number,team,espn_id};
   if(seen.has(number)&&JSON.stringify(seen.get(number))!==JSON.stringify(item))errors.push(`Conflicting pick ${number}`);
   else if(!seen.has(number)){picks.push(item);seen.set(number,item);}
  }
  picks.sort((a,b)=>a.number-b.number);
  for(let i=0;i<picks.length;i++)if(picks[i].number!==i+1)errors.push(`Missing pick ${i+1}; scroll draft history into view or adapt the row selector.`);
  if(new Set(picks.map(p=>p.espn_id)).size!==picks.length)errors.push('Duplicate drafted player');
  const timer=doc.querySelector(config.timerSelector||'[data-testid="draft-timer"], .draft-timer')?.textContent?.trim()||null;
  const available=Array.from(doc.querySelectorAll('[data-player-id][data-available="true"]')).map(e=>e.getAttribute('data-player-id'));
  return {picks,timer,available,current_pick:picks.length+1,observed_rows:rows.length,errors,compatible:rows.length>0&&errors.length===0};
 }
 function parseSettings(doc){
  // Evidence capture, not a guessed scoring import. User verifies these against local settings.
  return Array.from(doc.querySelectorAll('tr')).map(row=>Array.from(row.querySelectorAll('th,td')).map(e=>(e.textContent||'').trim())).filter(row=>row.length>=2&&row.some(x=>/scoring|roster|reception|touchdown|yards|waiver|playoff|draft|team/i.test(x)));
 }
 root.FantasyParser={parseDraft,parseSettings};
 if(typeof module!=='undefined')module.exports=root.FantasyParser;
})(typeof globalThis!=='undefined'?globalThis:this);
