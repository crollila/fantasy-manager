import {useEffect,useState} from 'react';

type Request=<T>(url:string,body?:unknown,method?:string)=>Promise<T>;
type Score={games:number;accuracy:number;brier:number;log_loss:number}|null;
type Bet={game_id:string;season:number;week:number;origin:string;matchup:string;team:string;model_probability:number;price:number;price_basis:string;edge:number;stake:number;profit:number;won:boolean;final:string;bankroll:number};
type Summary={bets:number;wins:number;staked:number;profit:number;roi:number|null;ending_bankroll:number|null;max_drawdown:number;average_edge:number|null;mid_priced:number};
type WeekH2H={season:number;week:number;origin:string;games:number;model:Score;market:Score;model_won:boolean};
type Source=Summary&{history:Bet[];priced_games:number;by_season:{season:number;origin:string;bets:number;wins:number;staked:number;profit:number;roi:number|null}[];threshold_sweep:(Summary&{threshold:number})[];head_to_head:{model:Score;market:Score;by_week:WeekH2H[];weeks_model_won:number;weeks:number}};
type Backtest={threshold:number;staking:string;origin:string;thresholds:number[];settings:{flat_stake:number;bankroll:number;kelly_fraction:number;kelly_cap:number};coverage:{live_games:number;walk_forward_games:number};sources:Record<string,Source>;notes:string[]};

const NAMES:Record<string,string>={kalshi:'Kalshi',polymarket:'Polymarket'};
const money=(v:number|null|undefined)=>v==null?'—':`${v<0?'−':''}$${Math.abs(v).toLocaleString(undefined,{maximumFractionDigits:0})}`;
const pct=(v:number|null|undefined,d=1)=>v==null?'—':`${(v*100).toFixed(d)}%`;
const cls=(v:number|null|undefined)=>v==null?'':v>=0?'pos-num':'neg-num';

function Equity({bets,label}:{bets:Bet[];label:string}){
 if(bets.length<2)return null;
 const series=bets.reduce<number[]>((acc,b)=>[...acc,(acc[acc.length-1]??0)+b.profit],[]);
 const lo=Math.min(0,...series),hi=Math.max(0,...series),W=640,H=170,pad=28;
 const x=(i:number)=>pad+i*(W-pad*2)/(series.length-1),y=(v:number)=>H-pad-(v-lo)/((hi-lo)||1)*(H-pad*2);
 return <div className="equity"><svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${label}: cumulative profit after ${series.length} bets, ending at ${money(series[series.length-1])}`}>
  <line className="axis" x1={pad} x2={W-pad} y1={H-pad} y2={H-pad}/><line className="zero" x1={pad} x2={W-pad} y1={y(0)} y2={y(0)}/>
  <polyline fill="none" stroke="var(--accent)" strokeWidth="1.8" points={series.map((v,i)=>`${x(i)},${y(v)}`).join(' ')}/>
  <text x={pad} y={14}>{money(hi)}</text><text x={pad} y={H-8}>{money(lo)}</text><text x={W-pad} y={H-8} textAnchor="end">{series.length} bets</text></svg></div>;
}

export function MarketBacktest({api,updatedAt}:{api:Request;updatedAt?:string}){
 const [threshold,setThreshold]=useState(0.05),[staking,setStaking]=useState('flat'),[origin,setOrigin]=useState('all');
 const [data,setData]=useState<Backtest|null>(null),[error,setError]=useState(''),[source,setSource]=useState('kalshi');
 useEffect(()=>{let alive=true;setError('');api<Backtest>(`/markets/backtest?threshold=${threshold}&staking=${staking}&origin=${encodeURIComponent(origin)}`).then(d=>{if(alive)setData(d);}).catch(e=>{if(alive)setError(String(e));});return()=>{alive=false;};},[threshold,staking,origin,updatedAt]);
 if(error)return <div className="alert error" role="alert">{error}</div>;
 if(!data)return <div className="empty">Running the backtest…</div>;
 const s=data.sources[source];
 const live=Object.entries(data.sources).map(([k,v])=>[k,v.head_to_head.by_week.filter(w=>w.origin==='live')] as const);
 return <>
  <p className="muted">If you had bet whenever our forecaster’s win probability beat the market’s kickoff price, what would have happened? Each “wins” contract costs its price and pays $1.</p>
  <section className="card"><h3>Weekly scoreboard: did we out-predict the markets?</h3>
   <p>After every finished week the app records Kalshi and Polymarket prices at kickoff and scores both against the results. Lower Brier score means better probabilities.</p>
   {live.some(([,w])=>w.length)?<table><thead><tr><th>Week</th>{live.map(([k])=><th key={k}>vs {NAMES[k]}</th>)}</tr></thead><tbody>{Array.from(new Set(live.flatMap(([,w])=>w.map(x=>`${x.season}-${x.week}`)))).sort((a,b)=>{const [sa,wa]=a.split('-').map(Number),[sb,wb]=b.split('-').map(Number);return sb-sa||wb-wa;}).map(key=><tr key={key}><td>{key.replace(/^(\d+)-/,'$1 · Week ')}</td>{live.map(([k,w])=>{const row=w.find(x=>`${x.season}-${x.week}`===key);return <td key={k}>{row&&row.model&&row.market?<><span className={`verdict ${row.model_won?'win':'loss'}`}>{row.model_won?'Beat market':'Market better'}</span> <small>ours {row.model.brier.toFixed(3)} · market {row.market.brier.toFixed(3)} · {row.games} games</small></>:'—'}</td>;})}</tr>)}</tbody></table>:<p>No live weeks with market prices yet. Prices are collected automatically on each refresh once a week’s games have finished.</p>}
   <div className="backtest-grid two">{Object.entries(data.sources).map(([k,v])=>{const h=v.head_to_head;return h.model&&h.market?<section className="card" key={k}><small>{NAMES[k].toUpperCase()} · {h.model.games} GAMES</small><h2>{h.weeks_model_won} of {h.weeks}</h2><p>weeks we beat {NAMES[k]} · Brier {h.model.brier.toFixed(3)} vs {h.market.brier.toFixed(3)} · picks {pct(h.model.accuracy,0)} vs {pct(h.market.accuracy,0)}</p></section>:null;})}</div>
   <small>Covers every game in the selected range below (walk-forward history and live weeks).</small>
  </section>
  <div className="toolbar"><label>Market<select value={source} onChange={e=>setSource(e.target.value)}>{Object.keys(data.sources).map(k=><option key={k} value={k}>{NAMES[k]}</option>)}</select></label>
   <label>Minimum edge<select value={threshold} onChange={e=>setThreshold(Number(e.target.value))}>{data.thresholds.map(t=><option key={t} value={t}>{t===0?'Any edge':`${(t*100).toFixed(1)} points`}</option>)}</select></label>
   <label>Stake<select value={staking} onChange={e=>setStaking(e.target.value)}><option value="flat">Flat ${data.settings.flat_stake} per bet</option><option value="kelly">Quarter Kelly, ${data.settings.bankroll} bankroll</option></select></label>
   <label>Games<select value={origin} onChange={e=>setOrigin(e.target.value)}><option value="all">All (history + live)</option><option value="live">Live forecasts only ({data.coverage.live_games})</option><option value="walk-forward">Walk-forward history only ({data.coverage.walk_forward_games})</option></select></label></div>
  {!s||!s.priced_games?<div className="welcome">No {NAMES[source]} prices are available for these games yet.</div>:<>
  <div className="backtest-grid">
   <section className="card"><small>PROFIT</small><h2 className={cls(s.profit)}>{money(s.profit)}</h2><p>on {money(s.staked)} staked</p></section>
   <section className="card"><small>RETURN ON STAKE</small><h2 className={cls(s.roi)}>{pct(s.roi)}</h2><p>{staking==='kelly'?`bankroll ${money(data.settings.bankroll)} → ${money(s.ending_bankroll)}`:'per dollar wagered'}</p></section>
   <section className="card"><small>BETS</small><h2>{s.wins}–{s.bets-s.wins}</h2><p>{s.bets} bets on {s.priced_games} priced games · avg edge {pct(s.average_edge)}</p></section>
   <section className="card"><small>WORST DRAWDOWN</small><h2>{money(s.max_drawdown)}</h2><p>largest fall from a peak</p></section>
  </div>
  <section className="card"><h3>Cumulative profit, bet by bet</h3><Equity bets={s.history} label={NAMES[source]}/>
   <table><thead><tr><th>Season</th><th>Source of forecasts</th><th>Bets</th><th>Won</th><th>Staked</th><th>Profit</th><th>Return</th></tr></thead><tbody>{s.by_season.map(r=><tr key={`${r.season}${r.origin}`}><td>{r.season}</td><td>{r.origin==='live'?'Live (saved before kickoff)':'Walk-forward backtest'}</td><td>{r.bets}</td><td>{r.wins}</td><td>{money(r.staked)}</td><td className={cls(r.profit)}>{money(r.profit)}</td><td className={cls(r.roi)}>{pct(r.roi)}</td></tr>)}</tbody></table></section>
  <section className="card table-scroll"><h3>How the edge threshold changes the result</h3><table><thead><tr><th>Minimum edge</th><th>Bets</th><th>Won</th><th>Profit</th><th>Return</th><th>Drawdown</th></tr></thead><tbody>{s.threshold_sweep.map(r=><tr key={r.threshold}><td>{r.threshold===0?'Any':`${(r.threshold*100).toFixed(1)} pts`}</td><td>{r.bets}</td><td>{r.wins}</td><td className={cls(r.profit)}>{money(r.profit)}</td><td className={cls(r.roi)}>{pct(r.roi)}</td><td>{money(r.max_drawdown)}</td></tr>)}</tbody></table></section>
  <section className="card table-scroll"><details><summary>Every bet ({s.history.length})</summary><table><thead><tr><th>Week</th><th>Game</th><th>Bet on</th><th>Ours</th><th>Price</th><th>Edge</th><th>Stake</th><th>Final</th><th>Profit</th></tr></thead><tbody>{[...s.history].reverse().map(b=><tr key={b.game_id}><td>{b.season} W{b.week}</td><td>{b.matchup}</td><td>{b.team}</td><td>{pct(b.model_probability,0)}</td><td>{(b.price*100).toFixed(0)}¢{b.price_basis==='mid'?'*':''}</td><td>{pct(b.edge)}</td><td>{money(b.stake)}</td><td>{b.final}</td><td className={cls(b.profit)}>{money(b.profit)}</td></tr>)}</tbody></table><small>* priced at the mid/last trade (no order book recorded). Finals are away–home.</small></details></section>
  </>}
  <div className="warning">Hypothetical results. No orders are placed. Past performance — especially on a few hundred games — does not guarantee future returns.</div>
  {data.notes.map(n=><small className="block" key={n}>{n}</small>)}
 </>;
}
