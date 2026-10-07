import {Fragment,useEffect,useState} from 'react';
import {TeamLogo} from './TeamLogo';

type Request=<T>(url:string,body?:unknown,method?:string)=>Promise<T>;
type Recent={week:number;opponent:string;home:boolean;points_for:number;points_against:number;result:string};
type Team={team:string;rank:number;previous_rank:number|null;change:number|null;rating:number;offense:number;defense:number;win_vs_average:number;elo:number|null;srs:number|null;wins:number;losses:number;ties:number;points_for:number;points_against:number;recent:Recent[];next_game:{week:number;opponent:string;home:boolean;kickoff:string|null}|null;history:{week:number;rank:number;rating:number}[]};
type Rankings={season:number;week:number;model_id:string;generated_at:string;teams:Team[];method:string;status?:string};

const signed=(v:number,d=1)=>`${v>=0?'+':''}${v.toFixed(d)}`;

function Spark({history}:{history:Team['history']}){
 if(history.length<2)return null;
 const w=90,h=26,x=(i:number)=>i*(w/(history.length-1)),y=(r:number)=>((r-1)/31)*h;
 return <svg className="rank-spark" width={w} height={h+4} viewBox={`0 -2 ${w} ${h+4}`} aria-label={`Rank by week: ${history.map(p=>p.rank).join(', ')}`}><polyline fill="none" points={history.map((p,i)=>`${x(i)},${y(p.rank)}`).join(' ')}/>{history.map((p,i)=><circle key={p.week} cx={x(i)} cy={y(p.rank)} r={i===history.length-1?2.6:1.6}><title>Week {p.week}: #{p.rank}</title></circle>)}</svg>;
}

export function PowerRankings({api,updatedAt}:{api:Request;updatedAt?:string}){
 const [data,setData]=useState<Rankings|null>(null),[error,setError]=useState(''),[open,setOpen]=useState<string|null>(null);
 useEffect(()=>{let alive=true;api<Rankings>('/power-rankings').then(d=>{if(alive)setData(d);}).catch(e=>{if(alive)setError(String(e));});return()=>{alive=false;};},[updatedAt]);
 if(error)return <div className="alert error" role="alert">{error}</div>;
 if(!data)return <div className="empty">Ranking all 32 teams…</div>;
 if(!data.teams.length)return <div className="welcome">{data.status??'Power rankings appear after the forecasting engine has loaded.'}</div>;
 const max=Math.max(...data.teams.map(t=>Math.abs(t.rating)));
 return <>
  <p className="muted">{data.season} · entering week {data.week}. Rating = points the model expects each team to beat an average team by on a neutral field.</p>
  <section className="card table-scroll"><table className="power-table"><thead><tr><th>#</th><th>Team</th><th>Record</th><th>Rating</th><th>Offense</th><th>Defense</th><th>Beats avg. team</th><th>Trend</th><th>Next</th></tr></thead><tbody>
  {data.teams.map(t=><Fragment key={t.team}><tr className="power-row" onClick={()=>setOpen(open===t.team?null:t.team)}>
   <td className="rank-cell"><b>{t.rank}</b>{t.change!=null&&t.change!==0&&<small className={t.change>0?'up':'down'}>{t.change>0?'▲':'▼'}{Math.abs(t.change)}</small>}</td>
   <td><span className="team-cell"><TeamLogo team={t.team} size={26}/><b>{t.team}</b></span></td>
   <td>{t.wins}–{t.losses}{t.ties?`–${t.ties}`:''}</td>
   <td><div className="rating-cell"><span className={`rating-bar ${t.rating>=0?'pos':'neg'}`} style={{width:`${Math.abs(t.rating)/max*50}%`,[t.rating>=0?'left':'right']:'50%'}}/><b>{signed(t.rating)}</b></div></td>
   <td>{signed(t.offense)}</td><td>{signed(t.defense)}</td><td>{(t.win_vs_average*100).toFixed(0)}%</td>
   <td><Spark history={t.history}/></td>
   <td>{t.next_game?<>{t.next_game.week!==data.week&&<small>Bye · W{t.next_game.week} </small>}{t.next_game.home?'vs':'@'} {t.next_game.opponent}</>:'—'}</td></tr>
   {open===t.team&&<tr className="power-detail"><td colSpan={9}><div className="power-detail-grid"><div><small>POINTS FOR / AGAINST</small><p>{t.points_for} / {t.points_against} ({signed(t.points_for-t.points_against,0)})</p></div><div><small>ELO · MARGIN RATING</small><p>{t.elo??'—'} · {t.srs==null?'—':signed(t.srs)}</p></div><div><small>RANK BY WEEK</small><p>{t.history.map(h=>`W${h.week} #${h.rank}`).join(' · ')}</p></div><div><small>RECENT GAMES</small><p>{t.recent.length?t.recent.map(r=>`W${r.week} ${r.result} ${r.points_for}–${r.points_against} ${r.home?'vs':'@'} ${r.opponent}`).join(' · '):'No games yet'}</p></div></div></td></tr>}</Fragment>)}
  </tbody></table></section>
  <small className="block">{data.method} Ranks for earlier weeks use the current model with each week’s pregame team data. Model {data.model_id}.</small>
 </>;
}
