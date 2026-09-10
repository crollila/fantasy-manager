'use strict';
function failure(code,message){return Object.assign(new Error(message),{code});}
function allowedEspnUrl(value) {
  try {const u=new URL(value);return u.protocol==='https:'&&['espn.com','go.com','disney.com','mydisney.com'].some(d=>u.hostname===d||u.hostname.endsWith('.'+d));} catch{return false;}
}
async function readLeague(fetch,team,week) {
  const url=new URL(`https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/${team.season}/segments/0/leagues/${team.league_id}`);
  for(const view of ['mSettings','mTeam','mRoster','mMatchup','mStatus'])url.searchParams.append('view',view);
  url.searchParams.set('scoringPeriodId',String(week));
  let response;
  try { response=await fetch(url.href,{credentials:'include',headers:{Accept:'application/json'},signal:AbortSignal.timeout(30000)}); }
  catch {throw failure('network','Could not reach ESPN. Your saved team is still available; check your connection and retry.');}
  if(response.status===401)throw failure('sign_in_required','ESPN has expired or rejected this sign-in. Reconnect ESPN once, then retry your team.');
  if(response.status===403)throw failure('access_denied','ESPN refused access to this league. Open it in the ESPN window to verify the account and complete any security prompt, then choose Retry import.');
  if(response.status===429)throw failure('rate_limited','ESPN is limiting requests. Wait a few minutes and retry; signing in again is not required.');
  if(response.status===404)throw failure('league_unavailable','ESPN could not find this league for the selected season. Check its My Team link.');
  if(!response.ok)throw failure('provider_unavailable','ESPN is temporarily unavailable. Your saved league remains available.');
  let snapshot;
  try {snapshot=await response.json();} catch {throw failure('unexpected_response','ESPN returned a sign-in or security page instead of league data. Open ESPN and complete that page, then choose Retry import.');}
  if(!snapshot||typeof snapshot!=='object'||!Array.isArray(snapshot.teams))throw failure('invalid_snapshot','ESPN did not return a complete league snapshot. Retry import after the My Team page finishes loading.');
  return snapshot;
}
module.exports={readLeague,allowedEspnUrl,failure};
