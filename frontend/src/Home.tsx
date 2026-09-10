import {useEffect,useState} from 'react';
import {Lineup} from './Lineup';
import {Intelligence,Research} from './Intelligence';

type Request=<T>(url:string,body?:unknown,method?:string)=>Promise<T>;
type League={id:string;name:string;season:number;my_team:number;team_names:string[];source:string;mode:string};
type Connection={url:string;name:string;league_id:string};
type SyncIssue={league_id:string;code:string;message:string};
type SessionStatus={hasSession:boolean;remembered:boolean;warning:string|null};
type ConnectionEvent={league_id?:string;state:string;code?:string;message:string;last_success?:string};
type SyncResult={cancelled?:boolean;league?:League};
declare global {interface Window {fantasyDesktop?:{
 status:()=>Promise<SessionStatus>;onConnection:(callback:(value:ConnectionEvent)=>void)=>()=>void;
 connect:(options:{url?:string;week:number})=>Promise<SyncResult>;
 refresh:(options:{url:string;week:number})=>Promise<SyncResult>;
 refreshSaved:()=>Promise<{synced:number;errors:SyncIssue[]}>;connections:()=>Promise<Connection[]>;openData:()=>Promise<void>;
}}}

export function Home({api,onAdvanced}:{api:Request;onAdvanced:()=>void}){
 const [sessionStatus,setSessionStatus]=useState<SessionStatus|null>(null),[connectionEvents,setConnectionEvents]=useState<Record<string,ConnectionEvent>>({}),[teamBusy,setTeamBusy]=useState(false);
 const [view,setView]=useState('My teams'),[research,setResearch]=useState<Research|null>(null),[researchBusy,setResearchBusy]=useState(false),[researchJob,setResearchJob]=useState('');
 const [leagues,setLeagues]=useState<League[]>([]),[connections,setConnections]=useState<Connection[]>([]);
 const [selected,setSelected]=useState(''),[error,setError]=useState(''),[busy,setBusy]=useState(false),[ready,setReady]=useState(false);
 const [adding,setAdding]=useState(false),[url,setUrl]=useState(''),[notice,setNotice]=useState('');
 async function reload(){const b=await api<{leagues:League[]}>('/bootstrap');const ls=b.leagues.filter(l=>l.source==='ESPN league snapshot');setLeagues(ls);const cs=await api<Connection[]>('/connections');setConnections(cs);setSelected(old=>old||ls[0]?.id||cs[0]?.league_id||'');setReady(true);}
 useEffect(()=>{reload().catch(e=>setError(String(e)));const desktop=window.fantasyDesktop;if(!desktop)return;desktop.status().then(setSessionStatus).catch(()=>{});return desktop.onConnection(value=>{if(value.league_id)setConnectionEvents(old=>({...old,[value.league_id!]:value}));desktop.status().then(setSessionStatus).catch(()=>{});});},[]);
 async function refreshTeams(){if(!window.fantasyDesktop||teamBusy)return;setTeamBusy(true);try{const sync=await window.fantasyDesktop.refreshSaved();await reload();setSessionStatus(await window.fantasyDesktop.status());setNotice(`${sync.synced} teams refreshed.${sync.errors.length?' '+sync.errors.map(e=>e.message).join(' '):''}`);}catch(e){setError(String(e));}finally{setTeamBusy(false);}}
 async function refreshResearch(){setResearchBusy(true);try{const j=await api<{id:string;status:string}>('/intelligence/refresh',{});setResearch(await api<Research>('/intelligence'));if(j.status==='complete')setResearchBusy(false);else setResearchJob(j.id);}catch(e){setResearchBusy(false);setError(String(e));}}
 useEffect(()=>{if(ready)refreshResearch();},[ready]);
 useEffect(()=>{if(!ready)return;const timer=setInterval(()=>api<Research>('/intelligence').then(setResearch).catch(()=>{}),30000);return()=>clearInterval(timer);},[ready]);
 useEffect(()=>{if(!researchJob)return;let alive=true;const timer=setInterval(async()=>{try{const j=await api<{status:string;error?:string}>(`/jobs/${researchJob}`);const r=await api<Research>('/intelligence');if(!alive)return;setResearch(r);if(['complete','error'].includes(j.status)){clearInterval(timer);if(j.error)setError(j.error);if(j.status==='complete')await refreshTeams();setResearchJob('');setResearchBusy(false);}}catch(e){if(alive){clearInterval(timer);setResearchJob('');setResearchBusy(false);setError(String(e));}}},3000);return()=>{alive=false;clearInterval(timer);};},[researchJob]);
 async function connect(value?:string){
  setBusy(true);setError('');setNotice('');
  try{
   if(value&&!connections.some(c=>c.url===value))await api('/connections',{url:value,name:'My ESPN team'});
   let result:SyncResult;
   if(window.fantasyDesktop)result=await window.fantasyDesktop.connect({url:value,week:research?.week??1});
   else {
    if(!value)throw new Error('Paste your ESPN My Team link below. The Windows application also supports private leagues through its ESPN sign-in window.');
    const u=new URL(value);result=await api('/espn/sync',{league_id:u.searchParams.get('leagueId'),my_team_id:Number(u.searchParams.get('teamId')),season:Number(u.searchParams.get('seasonId')||new Date().getFullYear()),week:research?.week??1});
   }
   await reload();if(window.fantasyDesktop)setSessionStatus(await window.fantasyDesktop.status());if(result.league){setSelected(result.league.id);setNotice(`${result.league.team_names[result.league.my_team]} connected.`);setAdding(false);setUrl('');}
  }catch(e){setError(String(e));await reload();}finally{setBusy(false);}
 }
 const league=leagues.find(l=>l.id===selected),connection=connections.find(c=>c.league_id===selected);
 const cards=[...connections,...leagues.filter(l=>!connections.some(c=>c.league_id===l.id)).map(l=>({league_id:l.id,name:l.team_names[l.my_team],url:''}))];
 async function refresh(week:number){
  if(!connection?.url)return;
  if(window.fantasyDesktop){await window.fantasyDesktop.refresh({url:connection.url,week});return;}
  const u=new URL(connection.url);await api('/espn/sync',{league_id:selected,my_team_id:Number(u.searchParams.get('teamId')),season:Number(u.searchParams.get('seasonId')||league?.season),week});
 }
 return <div className="home"><div className="home-top"><div className="brand"><strong>Fantasy Manager</strong><small>Football research &amp; lineup decisions</small></div><div className="home-actions"><span className="privacy-dot">{sessionStatus?.hasSession?'ESPN session saved':'Teams saved on this computer'}</span><button onClick={onAdvanced}>Advanced tools</button></div></div>
 <div className="home-content"><div className="home-tabs">{['My teams','NFL games','Accuracy','Data & evidence'].map(v=><button key={v} className={view===v?'selected':''} onClick={()=>setView(v)}>{v}</button>)}</div><header><div><div className="eyebrow">FANTASY FOOTBALL</div><h1>{view==='My teams'?'Your weekly lineup':view==='NFL games'?'NFL game forecasts':view==='Accuracy'?'Forecast accuracy':'Data & methodology'}</h1><p className="muted">{view==='My teams'?'Your teams, injury updates and a clear lineup recommendation.':'Updated on app open. Every forecast is saved before its game begins.'}</p></div><button className="primary" disabled={busy} onClick={()=>{setAdding(!adding);setView('My teams');}}>Manage ESPN</button></header>
 {sessionStatus?.warning&&<div className="alert" role="status">{sessionStatus.warning}</div>}{error&&<div className="alert error" role="alert">{error}<button onClick={()=>setError('')}>Dismiss</button></div>}{notice&&<div className="alert" role="status">{notice}</div>}
 {view!=='My teams'&&<Intelligence report={research} view={view} busy={researchBusy} onRefresh={refreshResearch}/>}
 {view==='My teams'&&<>
 {researchBusy&&<div className="alert" role="status">{research?.refresh?.stage||'Refreshing predictions and checking completed games…'}</div>}
 {adding&&<section className="card connect-card"><h3>Bring your team over from ESPN</h3><p>Your teams and ESPN session stay on this computer. Open My Team in the ESPN window to add a league; sign in only if ESPN asks. You keep control of lineup changes in ESPN.</p>{window.fantasyDesktop&&<button className="primary" disabled={busy} onClick={()=>connect()}>{busy?'Finish connecting in the ESPN window…':'Open ESPN'}</button>}<details open={!window.fantasyDesktop}><summary>I have a My Team link</summary><div className="toolbar"><input aria-label="ESPN My Team link" placeholder="Paste the full ESPN My Team link" value={url} onChange={e=>setUrl(e.target.value)}/><button disabled={busy||!url} onClick={()=>connect(url)}>Connect this team</button></div></details><small>ESPN sign-in stays in the app’s separate ESPN browser session. Remembered sign-in is encrypted with Windows on this computer. No passwords or session data are sent to GitHub.</small></section>}
 <div className="team-toolbar"><h3>Your teams</h3>{window.fantasyDesktop&&cards.length>0&&<button disabled={teamBusy||busy} onClick={refreshTeams}>{teamBusy?'Refreshing teams…':'Refresh teams'}</button>}</div><div className="league-cards" aria-label="Your teams">{cards.map(c=>{const l=leagues.find(l=>l.id===c.league_id);return <button className={`league-card ${selected===c.league_id?'selected':''}`} key={c.league_id} onClick={()=>setSelected(c.league_id)}><small>{connectionEvents[c.league_id]?.state==='connected'?'UP TO DATE':connectionEvents[c.league_id]?.state==='sign_in_required'?'RECONNECT NEEDED':l?'SAVED TEAM':'SAVED LINK'}</small><strong>{l?.team_names[l.my_team]||c.name}</strong><span>{l?.name||'ESPN Fantasy Football'}</span></button>;})}</div>
 {!ready&&!error&&<div className="empty">Opening your saved teams…</div>}
 {ready&&!cards.length&&<section className="welcome"><div className="welcome-icon">01</div><h2>Your best lineup starts here.</h2><p>Connect an ESPN team to compare every starter and bench player.<br/>We’ll account for your scoring rules, available projections and injury reports.</p><button className="primary" disabled={busy} onClick={()=>{setAdding(true);if(window.fantasyDesktop)connect();}}>Connect my first team</button><small>Prefer to explore first? Open Advanced tools for a clearly labeled demo.</small></section>}
 {connectionEvents[selected]&&connectionEvents[selected].state!=='connected'&&<div className="alert" role="status"><span>{connectionEvents[selected].message}</span>{connection&&<button disabled={busy} onClick={()=>connect(connection.url)}>Open ESPN</button>}</div>}{connection&&!league&&<section className="welcome"><h2>Your team link is saved.</h2><p>Import its scoring rules and roster from ESPN. If ESPN asks you to sign in, finish that step and choose <b>Retry import</b> at the top.</p><button className="primary" disabled={busy} onClick={()=>connect(connection.url)}>{busy?'Waiting for ESPN…':`Import ${connection.name}`}</button></section>}
 {league&&<Lineup key={selected} api={api} leagueId={selected} beforeAnalyze={refresh}/>}
 </>}
 <footer>Estimates, not guarantees. Review late injury news before kickoff.{window.fantasyDesktop&&<button className="text-button" onClick={()=>window.fantasyDesktop?.openData()}>Open local data folder</button>}</footer></div></div>;
}
