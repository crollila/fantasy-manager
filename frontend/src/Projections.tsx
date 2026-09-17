import {useEffect,useMemo,useState} from 'react';
type Request=<T>(url:string,body?:unknown,method?:string)=>Promise<T>;
type Row={id:string;name:string;position:string;team:string|null;opponent:string|null;mean:number;p10:number;p90:number;boom:number|null;bust:number|null;play_probability:number|null;injury_status:string|null;espn_projection:number|null;actual:number|null};
type Board={season?:number;week?:number;scoring:string;players:Row[]};
const POSITIONS=['All','QB','RB','WR','TE','K','DST'];
// Kept between tab switches, so the table is never blank while a newer copy loads.
let cached:Board|null=null;

export function Projections({api,updatedAt}:{api:Request;updatedAt?:string}){
 const [board,setBoard]=useState<Board|null>(cached),[position,setPosition]=useState('All'),[search,setSearch]=useState(''),[error,setError]=useState('');
 useEffect(()=>{api<Board>('/projections').then(b=>{cached=b;setBoard(b);}).catch(e=>setError(String(e)));},[updatedAt]);
 const rows=useMemo(()=>(board?.players??[]).filter(p=>(position==='All'||p.position===position)&&p.name.toLowerCase().includes(search.toLowerCase())),[board,position,search]);
 const graded=rows.filter(p=>p.actual!=null);
 return <section className="card table-scroll"><h3>Every player · Week {board?.week??'—'}</h3><p className="muted">{board?.scoring} Saved before kickoff and held until the next refresh replaces them.</p>
  <div className="toolbar"><label>Position<select value={position} onChange={e=>setPosition(e.target.value)}>{POSITIONS.map(p=><option key={p}>{p}</option>)}</select></label><label>Player<input aria-label="Search players" placeholder="Search by name" value={search} onChange={e=>setSearch(e.target.value)}/></label></div>
  {error&&<div className="alert error" role="alert">{error}</div>}
  {graded.length>0&&<p>{graded.length} finished · average miss <b>{(graded.reduce((s,p)=>s+Math.abs(p.actual!-p.mean),0)/graded.length).toFixed(1)}</b> points.</p>}
  <table><thead><tr><th>#</th><th>Player</th><th>Status</th><th>Projection</th><th>Likely range</th><th>ESPN</th><th>Boom</th><th>Bust</th><th>Actual</th></tr></thead><tbody>{rows.slice(0,400).map((p,i)=><tr key={p.id}><td>{i+1}</td><td>{p.name}<small> {p.position} · {p.team??'—'}{p.opponent?` vs ${p.opponent}`:''}</small></td><td><small>{(p.injury_status??'unknown').replaceAll('_',' ').toLowerCase()}</small></td><td><b>{p.mean.toFixed(1)}</b></td><td>{p.p10.toFixed(1)}–{p.p90.toFixed(1)}</td><td>{p.espn_projection?.toFixed(1)??'—'}</td><td>{p.boom==null?'—':`${(p.boom*100).toFixed(0)}%`}</td><td>{p.bust==null?'—':`${(p.bust*100).toFixed(0)}%`}</td><td>{p.actual?.toFixed(1)??'—'}</td></tr>)}</tbody></table>
  {board&&!board.players.length&&<div className="empty">No projections saved yet. They are created by “Refresh data &amp; results”.</div>}{rows.length>400&&<small>Showing the top 400 of {rows.length}. Filter or search to narrow.</small>}
  <small>ESPN column: ESPN’s projected stat line for players rostered in your leagues, rescored under the same default rules so the two numbers are comparable.</small></section>;
}
