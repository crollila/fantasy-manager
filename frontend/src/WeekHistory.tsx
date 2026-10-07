import {useEffect,useState} from 'react';
import {TeamLogo} from './TeamLogo';

type Request=<T>(url:string,body?:unknown,method?:string)=>Promise<T>;
export type WeekOption={season:number;week:number;games:number;graded:number;correct:number};
type MarketPrice={home_prob_norm:number;home_prob:number|null;away_prob:number|null;minutes_before_kickoff:number|null;url?:string};
type HistoryForecast={home_score:number;away_score:number;pick:string;home_win_probability:number;away_win_probability:number;home_win_probability_norm:number;pick_win_probability:number;opening_home_win_probability:number;model_version:string;saved_at:string;versions:number;market_home_margin:number|null};
type HistoryGame={game_id:string;home_team:string;away_team:string;kickoff:string|null;forecast:HistoryForecast|null;final:{home_score:number;away_score:number}|null;markets:Record<string,MarketPrice>|null;pick_result?:string;score_error?:number;winner?:string};
type Score={games:number;accuracy:number;brier:number;log_loss:number}|null;
type WeekData={season:number;week:number;games:HistoryGame[];note:string;summary:{games:number;forecasts:number;finished:number;wins:number;losses:number;ties:number;score_mae:number|null;models:string[];probability:{model:Score;kalshi?:{market:Score;model:Score};polymarket?:{market:Score;model:Score}}}};

const pct=(v:number|null|undefined)=>v==null?'—':`${(v*100).toFixed(0)}%`;
const one=(v:number|null|undefined)=>v==null?'—':v.toFixed(1);
const SOURCES:[string,string][]=[['kalshi','Kalshi'],['polymarket','Polymarket']];
const modelName=(v:string)=>v.startsWith('engine:')?`Engine refit ${v.slice(-16,-8).replace(/(\d{4})(\d{2})(\d{2})/,'$1-$2-$3')}`:`Legacy model ${v}`;

function ProbabilityBar({away,home}:{away:number;home:number}){
 return <div className="prob-bar" aria-hidden="true"><span style={{width:`${away*100}%`}}/><span style={{width:`${home*100}%`}}/></div>;
}

export function WeekHistory({api,season,week}:{api:Request;season:number;week:number}){
 const [data,setData]=useState<WeekData|null>(null),[error,setError]=useState(''),[layout,setLayout]=useState<'cards'|'table'>('cards');
 useEffect(()=>{let alive=true;setData(null);setError('');api<WeekData>(`/intelligence/weeks/${season}/${week}`).then(d=>{if(alive)setData(d);}).catch(e=>{if(alive)setError(String(e));});return()=>{alive=false;};},[season,week]);
 if(error)return <div className="alert error" role="alert">{error}</div>;
 if(!data)return <div className="empty">Loading week {week}…</div>;
 const s=data.summary,p=s.probability;
 const decided=s.wins+s.losses;
 return <>
  <div className="accuracy-grid week-summary">
   <section className="card"><small>WINNER PICKS</small><h2>{decided?`${s.wins}–${s.losses}`:'—'}</h2><p>{decided?`${pct(s.wins/decided)} correct`:'Games not finished yet'}{s.ties?` · ${s.ties} tie`:''}</p></section>
   <section className="card"><small>PROBABILITY ERROR</small><h2>{p.model?p.model.brier.toFixed(3):'—'}</h2><p>Brier score (lower is better; always guessing 50% scores 0.250) · log loss {p.model?p.model.log_loss.toFixed(3):'—'}</p></section>
   <section className="card"><small>SCORE ERROR</small><h2>{s.score_mae==null?'—':`${one(s.score_mae)} pts`}</h2><p>Average miss per team score · {s.forecasts} of {s.games} games forecast</p></section>
  </div>
  {SOURCES.some(([k])=>p[k as 'kalshi'])&&<section className="card"><h3>Against the prediction markets this week</h3><table><thead><tr><th>Market</th><th>Games</th><th>Market picks correct</th><th>Our picks correct</th><th>Market Brier</th><th>Our Brier</th></tr></thead><tbody>{SOURCES.map(([k,label])=>{const m=p[k as 'kalshi'];return m&&m.market&&m.model?<tr key={k}><td>{label}</td><td>{m.market.games}</td><td>{pct(m.market.accuracy)}</td><td>{pct(m.model.accuracy)}</td><td>{m.market.brier.toFixed(3)}</td><td><b>{m.model.brier.toFixed(3)}</b></td></tr>:null;})}</tbody></table><small>Market probability is the price of each team’s “wins” contract at kickoff, normalised so both sides add to 100%. Same games only.</small></section>}
  <div className="toolbar week-layout"><span className="muted">{s.models.length?`Forecasts from: ${s.models.map(modelName).join(', ')}`:''}</span><div className="segmented">{(['cards','table'] as const).map(v=><button key={v} className={layout===v?'selected':''} onClick={()=>setLayout(v)}>{v==='cards'?'Cards':'Table'}</button>)}</div></div>
  {layout==='cards'?<div className="game-grid">{data.games.map(g=>{const f=g.forecast,done=g.final!=null;const homeP=f?.home_win_probability_norm;return <div className={`game-card history-card ${g.pick_result==='W'?'hit':g.pick_result==='L'?'miss':''}`} key={g.game_id}>
   <div className="game-date">{g.kickoff?new Date(g.kickoff).toLocaleString(undefined,{weekday:'short',month:'short',day:'numeric',hour:'numeric',minute:'2-digit'}):''}<span>{g.pick_result==='W'?'✓ Correct pick':g.pick_result==='L'?'✗ Missed pick':g.pick_result==='T'?'Tie':done?'Final':'Upcoming'}</span></div>
   <div className="history-head"><span/><small>WIN %</small><small>PROJ</small><small>FINAL</small></div>
   {(['away','home'] as const).map(side=>{const team=side==='home'?g.home_team:g.away_team;const prob=homeP==null?null:side==='home'?homeP:1-homeP;const proj=f?(side==='home'?f.home_score:f.away_score):null;const fin=g.final?(side==='home'?g.final.home_score:g.final.away_score):null;const won=g.winner===team;return <div className={`history-row ${won?'winner':''}`} key={side}><strong><TeamLogo team={team} size={24}/>{team}{side==='home'&&<small> HOME</small>}</strong><span>{pct(prob)}</span><span>{one(proj)}</span><b>{fin??'—'}</b></div>;})}
   {f&&homeP!=null&&<ProbabilityBar away={1-homeP} home={homeP}/>}
   <div className="game-pick">{f?<><strong><TeamLogo team={f.pick} size={18}/>Pick: {f.pick}</strong><span>{pct(f.pick_win_probability)} to win</span></>:<span>No pregame forecast was saved for this game</span>}</div>
   {g.markets&&<div className="market-line">{SOURCES.filter(([k])=>g.markets?.[k]).map(([k,label])=>{const m=g.markets![k];return <span key={k}>{label}: {g.home_team} {pct(m.home_prob_norm)} · {g.away_team} {pct(1-m.home_prob_norm)}</span>;})}</div>}
   {f&&<small>Opened at {g.home_team} {pct(f.opening_home_win_probability)} · {f.versions} saved version{f.versions===1?'':'s'} · last {new Date(f.saved_at).toLocaleString(undefined,{month:'short',day:'numeric',hour:'numeric',minute:'2-digit'})}</small>}
  </div>;})}</div>
  :<section className="card table-scroll"><table><thead><tr><th>Game</th><th>Our pick</th><th>Our win %</th>{SOURCES.map(([k,l])=><th key={k}>{l} (same team)</th>)}<th>Projected</th><th>Final</th><th>Result</th></tr></thead><tbody>{data.games.map(g=>{const f=g.forecast;const pickHome=f?f.pick===g.home_team:false;return <tr key={g.game_id}><td>{g.away_team} at {g.home_team}</td><td>{f?.pick??'—'}</td><td>{f?pct(f.pick_win_probability):'—'}</td>{SOURCES.map(([k])=>{const m=g.markets?.[k];return <td key={k}>{m&&f?pct(pickHome?m.home_prob_norm:1-m.home_prob_norm):'—'}</td>;})}<td>{f?`${one(f.away_score)}–${one(f.home_score)}`:'—'}</td><td>{g.final?`${g.final.away_score}–${g.final.home_score}`:'—'}</td><td>{g.pick_result==='W'?'✓':g.pick_result==='L'?'✗':g.pick_result??'—'}</td></tr>;})}</tbody></table><small>Scores are listed away–home.</small></section>}
  <small className="block">{data.note}</small>
 </>;
}
