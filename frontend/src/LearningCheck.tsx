import {useEffect,useState} from 'react';

type Request=<T>(url:string,body?:unknown,method?:string)=>Promise<T>;
type Score={games:number;log_loss:number;brier:number;accuracy:number;margin_mae:number}|null;
type Week={week:number;games:number;live_model:string;live:Score;no_refit:Score;no_new_results:Score;refit_probability_change:number;data_probability_change:number};
type Bucket={updated:Score;frozen:Score;mean_abs_prob_change:number};
type Check={season:number;weeks:Week[];season_scores?:Record<string,Score>;models_this_season?:{model_id:string;trained_through_week:number}[];status?:string;
 history?:{seasons:number[];method:string;pooled:Record<string,Bucket>}|null};

const pct=(v:number|undefined|null)=>v==null?'—':`${(v*100).toFixed(1)}%`;
const ll=(s:Score)=>s?s.log_loss.toFixed(3):'—';

export function LearningCheck({api,updatedAt}:{api:Request;updatedAt?:string}){
 const [data,setData]=useState<Check|null>(null),[error,setError]=useState('');
 useEffect(()=>{api<Check>('/nfl/learning-check').then(setData).catch(e=>setError(String(e)));},[updatedAt]);
 if(error)return <div className="alert error" role="alert">Learning check unavailable: {error}</div>;
 if(!data)return <section className="card"><h3>Is the forecaster learning?</h3><p>Re-scoring this season’s weeks…</p></section>;
 const s=data.season_scores,h=data.history;
 const gain=s?.live&&s?.no_new_results?s.no_new_results.log_loss-s.live.log_loss:null;
 return <section className="card table-scroll"><h3>Is the forecaster learning from each week?</h3>
  <p>Every completed week is re-scored three ways, each using only models trained before that week: what the app actually did, the same thing without the weekly model refit, and a forecaster that never learns from new results (team ratings, form and quarterback stats frozen at week 1). Log loss measures probability quality; lower is better.</p>
  {data.weeks.length?<table><thead><tr><th>Week</th><th>Games</th><th>Live (what the app did)</th><th>Without weekly refit</th><th>Never learns from results</th><th>Refit moved odds by</th><th>New results moved odds by</th></tr></thead><tbody>{data.weeks.map(w=><tr key={w.week}><td>Week {w.week}</td><td>{w.games}</td><td><b>{ll(w.live)}</b> · {pct(w.live?.accuracy)}</td><td>{ll(w.no_refit)} · {pct(w.no_refit?.accuracy)}</td><td>{ll(w.no_new_results)} · {pct(w.no_new_results?.accuracy)}</td><td>{pct(w.refit_probability_change)}</td><td>{pct(w.data_probability_change)}</td></tr>)}
  {s&&<tr><td><b>Season</b></td><td>{s.live?.games}</td><td><b>{ll(s.live)}</b> · {pct(s.live?.accuracy)}</td><td>{ll(s.no_refit)} · {pct(s.no_refit?.accuracy)}</td><td>{ll(s.no_new_results)} · {pct(s.no_new_results?.accuracy)}</td><td/><td/></tr>}</tbody></table>:<p>{data.status??'No completed weeks yet.'}</p>}
  {gain!=null&&<p>{gain>0?<>So far this season, learning from results has improved log loss by <b>{gain.toFixed(3)}</b> versus a forecaster frozen at week 1.</>:<>So far this season the frozen forecaster scores as well as the live one ({Math.abs(gain).toFixed(3)} difference) — expected in the first weeks, when there are few new results to learn from.</>} The weekly refit itself changes odds by well under one point; most learning comes from updated team ratings and form.</p>}
  {h&&<><h4 style={{marginTop:20}}>Longer test: {h.seasons[0]}–{h.seasons[h.seasons.length-1]} seasons (out of sample)</h4><table><thead><tr><th>Part of season</th><th>Games</th><th>Learning from results</th><th>Frozen at week 1</th><th>Odds moved by</th></tr></thead><tbody>{Object.entries(h.pooled).map(([k,b])=><tr key={k}><td>{k}</td><td>{b.updated?.games}</td><td><b>{pct(b.updated?.accuracy)}</b> correct · log loss {ll(b.updated)}</td><td>{pct(b.frozen?.accuracy)} correct · log loss {ll(b.frozen)}</td><td>{pct(b.mean_abs_prob_change)}</td></tr>)}</tbody></table><small className="block">{h.method}</small></>}
  <small className="block">{data.models_this_season?.length??0} weekly refits this season. Small samples: a single week of 16 games can favour either version by chance; judge the trend across weeks.</small>
 </section>;
}
