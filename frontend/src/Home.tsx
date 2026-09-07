import {useEffect,useState} from 'react';
import {Lineup} from './Lineup';

type Request=<T>(url:string,body?:unknown,method?:string)=>Promise<T>;
type League={id:string;name:string;season:number;my_team:number;team_names:string[];source:string;mode:string};
type Connection={url:string;name:string;league_id:string};
type SyncResult={cancelled?:boolean;league?:League};
declare global {interface Window {fantasyDesktop?:{
 connect:(options:{url?:string;week:number})=>Promise<SyncResult>;
 refresh:(options:{url:string;week:number})=>Promise<SyncResult>;
 connections:()=>Promise<Connection[]>;openData:()=>Promise<void>;
}}}

export function Home({api,onAdvanced}:{api:Request;onAdvanced:()=>void}){
 const [leagues,setLeagues]=useState<League[]>([]),[connections,setConnections]=useState<Connection[]>([]);
 const [selected,setSelected]=useState(''),[error,setError]=useState(''),[busy,setBusy]=useState(false),[ready,setReady]=useState(false);
 const [adding,setAdding]=useState(false),[url,setUrl]=useState(''),[notice,setNotice]=useState('');
 async function reload(){const b=await api<{leagues:League[]}>('/bootstrap');const ls=b.leagues.filter(l=>l.source==='ESPN league snapshot');setLeagues(ls);const cs=await api<Connection[]>('/connections');setConnections(cs);setSelected(old=>old||ls[0]?.id||cs[0]?.league_id||'');setReady(true);}
 useEffect(()=>{reload().catch(e=>setError(String(e)));},[]);
 async function connect(value?:string){
  setBusy(true);setError('');setNotice('');
  try{
   if(value)await api('/connections',{url:value,name:'My ESPN team'});
   let result:SyncResult;
   if(window.fantasyDesktop)result=await window.fantasyDesktop.connect({url:value,week:1});
   else {
    if(!value)throw new Error('Paste your ESPN My Team link below. The Windows application also supports private leagues through its ESPN sign-in window.');
    const u=new URL(value);result=await api('/espn/sync',{league_id:u.searchParams.get('leagueId'),my_team_id:Number(u.searchParams.get('teamId')),season:Number(u.searchParams.get('seasonId')||new Date().getFullYear()),week:1});
   }
   await reload();if(result.league){setSelected(result.league.id);setNotice(`${result.league.team_names[result.league.my_team]} connected.`);setAdding(false);setUrl('');}
  }catch(e){setError(String(e));await reload();}finally{setBusy(false);}
 }
 const league=leagues.find(l=>l.id===selected),connection=connections.find(c=>c.league_id===selected);
 const cards=[...connections,...leagues.filter(l=>!connections.some(c=>c.league_id===l.id)).map(l=>({league_id:l.id,name:l.team_names[l.my_team],url:''}))];
 async function refresh(week:number){
  if(!connection?.url)return;
  if(window.fantasyDesktop){await window.fantasyDesktop.refresh({url:connection.url,week});return;}
  const u=new URL(connection.url);await api('/espn/sync',{league_id:selected,my_team_id:Number(u.searchParams.get('teamId')),season:Number(u.searchParams.get('seasonId')||league?.season),week});
 }
 return <div className="home"><div className="home-top"><div className="brand"><span>FM</span> Fantasy Manager</div><div className="home-actions"><span className="privacy-dot">Saved on this computer</span><button onClick={onAdvanced}>Advanced tools</button></div></div>
 <div className="home-content"><header><div><div className="eyebrow">YOUR WEEKLY GAME PLAN</div><h1>Make the right starts.</h1><p className="muted">Your teams, injury updates and a clear lineup recommendation.</p></div><button className="primary" disabled={busy} onClick={()=>setAdding(!adding)}>+ Connect ESPN</button></header>
 {error&&<div className="alert error" role="alert">{error}<button onClick={()=>setError('')}>Dismiss</button></div>}{notice&&<div className="alert" role="status">{notice}</div>}
 {adding&&<section className="card connect-card"><h3>Bring your team over from ESPN</h3><p>Open your team from ESPN’s Fantasy menu. We’ll import its scoring and roster automatically. You keep control of all lineup changes in ESPN.</p>{window.fantasyDesktop&&<button className="primary" disabled={busy} onClick={()=>connect()}>{busy?'Finish connecting in the ESPN window…':'Open ESPN & sign in'}</button>}<details open={!window.fantasyDesktop}><summary>I have a My Team link</summary><div className="toolbar"><input aria-label="ESPN My Team link" placeholder="Paste the full ESPN My Team link" value={url} onChange={e=>setUrl(e.target.value)}/><button disabled={busy||!url} onClick={()=>connect(url)}>Connect this team</button></div></details><small>ESPN sign-in stays in the app’s separate ESPN browser session. No passwords or cookies are copied into your project.</small></section>}
 <div className="league-cards" aria-label="Your teams">{cards.map(c=>{const l=leagues.find(l=>l.id===c.league_id);return <button className={`league-card ${selected===c.league_id?'selected':''}`} key={c.league_id} onClick={()=>setSelected(c.league_id)}><small>{l?'CONNECTED':'SIGN-IN NEEDED'}</small><strong>{l?.team_names[l.my_team]||c.name}</strong><span>{l?.name||'ESPN Fantasy Football'}</span></button>;})}</div>
 {!ready&&!error&&<div className="empty">Opening your saved teams…</div>}
 {ready&&!cards.length&&<section className="welcome"><div className="welcome-icon">01</div><h2>Your best lineup starts here.</h2><p>Connect an ESPN team to compare every starter and bench player.<br/>We’ll account for your scoring rules, available projections and injury reports.</p><button className="primary" disabled={busy} onClick={()=>{setAdding(true);if(window.fantasyDesktop)connect();}}>Connect my first team</button><small>Prefer to explore first? Open Advanced tools for a clearly labeled demo.</small></section>}
 {connection&&!league&&<section className="welcome"><h2>One sign-in, then you’re ready.</h2><p>Open this team in the ESPN connection window. Sign in if asked, then choose <b>Import this team</b> at the top.</p><button className="primary" disabled={busy} onClick={()=>connect(connection.url)}>{busy?'Waiting for ESPN…':`Connect ${connection.name}`}</button></section>}
 {league&&<Lineup key={selected} api={api} leagueId={selected} beforeAnalyze={refresh}/>}
 <footer>Estimates, not guarantees. Review late injury news before kickoff.{window.fantasyDesktop&&<button className="text-button" onClick={()=>window.fantasyDesktop?.openData()}>Open local data folder</button>}</footer></div></div>;
}
