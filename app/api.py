from __future__ import annotations
import asyncio
import os
import json
import secrets
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import Field
from app.domain import League, Pick, Player, Event, Strict, Snapshot
from app.storage import Store, ROOT, DATA, now
from app.demo import demo_players,demo_league
from app.projections import project_all,build_catalog,apply_events,ensemble_weight
from app.draft import quick_recommendations
from app.simulation import evaluate_candidates,mock_report
from app.data import refresh_sources,Cache,historical
from app.backtest import backtest
from app.espn import normalize_snapshot
from app.season import start_sit,waivers,trade

store = Store()
pool = ThreadPoolExecutor(max_workers=2,thread_name_prefix="fantasy-job")
jobs = {}
job_lock = threading.Lock()
refresh_lock = threading.Lock()
report_cache = {}
report_lock = threading.Lock()


def context(league_id):
    league = store.league(league_id)
    players = apply_events(store.players(league.season),store.events(),now())
    meta = store.catalog_meta(league.season)
    key = (league.model_dump_json(),meta["revision"],tuple(e.model_dump_json() for e in store.events()))
    with report_lock:
        if key not in report_cache:
            report_cache.clear() if len(report_cache)>12 else None
            report_cache[key] = project_all(players,league)
        reports = report_cache[key]
    draft = store.draft(league_id)
    picks = [Pick.model_validate(p) for p in draft["picks"]]
    if league.mode=="SEASON":
        from app.ownership import current_ownership
        picks,_=current_ownership(store,league,picks)
    return league,players,reports,picks,draft


def submit(kind,fn,league_id=None,fingerprint=None):
    with job_lock:
        if sum(j["status"] in ("queued","running") for j in jobs.values()) >= 3:
            raise ValueError("Background queue is full; wait for current work")
        job_id = uuid.uuid4().hex
        if len(jobs)>40:
            for key in list(jobs):
                if jobs[key]["status"] in ("complete","error"):
                    del jobs[key]
                    break
        jobs[job_id] = {"id":job_id,"kind":kind,"status":"queued","created":now(),"league_id":league_id,"fingerprint":fingerprint}
    def work():
        jobs[job_id]["status"] = "running"
        started = time.monotonic()
        try:
            result = fn()
            jobs[job_id].update(status="complete",result=result)
        except Exception as e:
            jobs[job_id].update(status="error",error=str(e))
        jobs[job_id]["seconds"] = round(time.monotonic()-started,2)
    pool.submit(work)
    return jobs[job_id].copy()


def refresh(season):
    with refresh_lock:
        return refresh_locked(season)


def refresh_locked(season):
    statuses = refresh_sources(season)
    players = build_catalog(season)
    # Preserve explicit mappings and market data across refreshes.
    old = {p.id:p for p in store.players(season)}
    for p in players:
        if p.id in old:
            previous = old[p.id]
            p.ids = previous.ids | p.ids
            for field in ("adp","adp_sd","espn_rank","ecr","market_stats","market_weight"):
                setattr(p,field,getattr(previous,field))
    for p in old.values():
        if p.position in ("K","DST") and p.id not in {q.id for q in players}:
            players.append(p)
    store.save_players(season,players)
    from app.news import refresh_news
    news=refresh_news()
    return {"news": {"status":news["status"],"items":len(news["items"])},"sources":statuses,"players":len(players),"updated":now()}


@asynccontextmanager
async def lifespan(app):
    if not store.leagues():
        demo = demo_league()
        demo.season = 2099
        store.save_players(2099,demo_players())
        store.save_league(demo)
    async def scheduled():
        while True:
            await asyncio.sleep(6*3600)
            for season in {l.season for l in store.leagues() if l.season != 2099}:
                try:
                    await asyncio.to_thread(refresh,season)
                except Exception as exc:
                    (DATA / "refresh-error.json").write_text(json.dumps({"at":now(),"error":str(exc)}))
    async def intelligence_schedule():
        while True:
            await asyncio.sleep(900)
            if os.environ.get('FANTASY_DISABLE_AUTO_REFRESH')=='1':continue
            try:
                from app.intelligence import refresh_intelligence
                await asyncio.to_thread(refresh_intelligence,store)
            except Exception:
                pass  # Coordinator persists the error and retains the previous forecast.
    intel_task=asyncio.create_task(intelligence_schedule())
    task = asyncio.create_task(scheduled())
    yield
    task.cancel()
    intel_task.cancel()


app = FastAPI(title="Fantasy Manager",version="0.5.0",lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=["127.0.0.1","localhost","testserver"])


@app.middleware("http")
async def local_auth(request:Request,call_next):
    origin = request.headers.get("origin","")
    allowed_origin = origin == f"http://127.0.0.1:{os.environ.get('FANTASY_PORT','8000')}" or origin in ("http://127.0.0.1:8000","http://localhost:8000","http://127.0.0.1:5173","http://localhost:5173")
    if request.url.path.startswith("/api/"):
        if origin and not allowed_origin and not origin.startswith("chrome-extension://"):
            return JSONResponse({"detail":"Origin not allowed"},403)
        if request.url.path not in ("/api/bootstrap","/api/health"):
            if not secrets.compare_digest(request.headers.get("x-local-token",""),store.token):
                return JSONResponse({"detail":"Local pairing token required"},401)
        elif origin.startswith("chrome-extension://"):
            return JSONResponse({"detail":"Pair manually using the token displayed in the local app"},403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api/") else "no-cache"
    return response


@app.exception_handler(ValueError)
async def value_error(request,exc):
    return JSONResponse({"detail":str(exc)},400)


@app.get("/api/health")
def health():
    return {"app":"fantasy-manager","version":"0.5.0","instance":os.environ.get("FANTASY_INSTANCE","")}


@app.get("/api/connections")
def connections():
    with store.connect() as c:
        row=c.execute("SELECT value FROM meta WHERE key='connections'").fetchone()
    return json.loads(row[0]) if row else []


class ConnectionRequest(Strict):
    url: str
    name: str = "My ESPN team"


@app.post("/api/connections")
def save_connection(body:ConnectionRequest):
    from urllib.parse import urlparse,parse_qs
    parsed=urlparse(body.url)
    query=parse_qs(parsed.query)
    if parsed.scheme!="https" or parsed.netloc!="fantasy.espn.com" or parsed.path!="/football/team" or not query.get("leagueId",[""])[0].isdigit() or not query.get("teamId",[""])[0].isdigit():
        raise ValueError("Paste the ESPN My Team URL")
    rows=connections()
    item={"url":body.url,"name":body.name,"league_id":query["leagueId"][0]}
    rows=[r for r in rows if r["league_id"]!=item["league_id"]]+[item]
    with store.connect() as c:c.execute("INSERT OR REPLACE INTO meta VALUES('connections',?)",(json.dumps(rows),))
    return rows


@app.get("/api/bootstrap")
def bootstrap():
    return {"token":store.token,"version":"0.5.0","leagues":[l.model_dump() for l in store.leagues()]}


@app.get("/api/leagues")
def leagues():
    return [l.model_dump() for l in store.leagues()]


@app.put("/api/leagues/{league_id}")
def put_league(league_id:str,body:League):
    if body.id != league_id:
        raise ValueError("Path and league ID differ")
    if body.season==2099 and body.source!="SYNTHETIC DEMO":
        raise ValueError("Season 2099 is reserved for the isolated synthetic demo")
    store.save_league(body)
    return body


@app.get("/api/leagues/{league_id}/state")
def state(league_id:str):
    league,players,reports,picks,draft = context(league_id)
    recs = quick_recommendations(players,reports,league,picks) if league.draft_type != "auction" and league.mode=="DRAFT" else []
    warnings = []
    provided={key for p in players for key in p.stats}
    missing={key for key,value in league.scoring.items() if value and key not in provided}
    if missing and players and not all(p.source=="SYNTHETIC DEMO" for p in players):
        warnings.append("No source components for scoring fields: "+", ".join(sorted(missing)))
    if not league.settings_verified:
        warnings.append("League settings have not been verified against ESPN")
    if any(p.source=="SYNTHETIC DEMO" for p in players):
        warnings.append("SYNTHETIC DEMO: fictional players and estimates")
    for pos,n in league.slots.items():
        if n and pos in ("K","DST") and not any(p.position==pos for p in players):
            warnings.append(f"Missing {pos} pool: import source-backed projections; complete draft simulation disabled")
    if league.slots.get("DST",0):
        warnings.append("Verify DST points/yards-allowed scoring: default template covers event components only")
    if any(p.adp is None for p in players):
        warnings.append("ADP missing for some players: survival unavailable; simulations fall back to model ranking")
    return {"league":league,"draft":draft,"players":reports,"recommendations":recs[:12],"catalog":store.catalog_meta(league.season),"warnings":warnings,"current_owner":league.owner(len(picks)+1) if league.draft_type!="auction" and len(picks)<league.teams*league.roster_size else None}


@app.post("/api/leagues/{league_id}/draft")
def update_draft(league_id:str,body:Snapshot,revision:int):
    league = store.league(league_id)
    return {"revision":store.save_draft(league,body.picks,body.source,revision)}


@app.post("/api/leagues/{league_id}/reset")
def reset_draft(league_id:str,body:Snapshot,revision:int):
    return {"revision":store.save_draft(store.league(league_id),body.picks,"explicit reset/replay",revision,True)}


@app.post("/api/leagues/{league_id}/espn")
def espn(league_id:str,body:dict):
    league = store.league(league_id)
    picks = normalize_snapshot(body,league,store.players(league.season))
    return {"revision":store.save_draft(league,picks,"ESPN browser DOM"),"count":len(picks)}


class SimulationRequest(Strict):
    simulations:int = Field(default=1000,ge=64,le=10000)
    candidates:int = Field(default=6,ge=1,le=12)
    seed:int = Field(default=42,ge=0,le=1000000)


@app.post("/api/leagues/{league_id}/simulate")
def simulate(league_id:str,body:SimulationRequest):
    league,players,reports,picks,draft = context(league_id)
    if not players:
        raise ValueError("Load projections first")
    if len(picks)>=league.teams*league.roster_size:
        raise ValueError("Draft is complete")
    candidates = quick_recommendations(players,reports,league,picks)[:body.candidates]
    fingerprint = {"draft_revision":draft["revision"],"catalog_revision":store.catalog_meta(league.season)["revision"],"settings":league.model_dump_json(),"events":[e.model_dump_json() for e in store.events()]}
    return submit("draft",lambda:evaluate_candidates(players,reports,league,picks,[r["id"] for r in candidates],body.simulations,body.seed),league_id,fingerprint)


@app.post("/api/leagues/{league_id}/mock")
def mock(league_id:str,body:SimulationRequest):
    league,players,reports,*_ = context(league_id)
    return submit("mock",lambda:mock_report(players,reports,league,body.simulations,body.seed),league_id)


@app.get("/api/jobs/{job_id}")
def job(job_id:str):
    if job_id not in jobs:
        raise HTTPException(404,"Job not found (jobs are transient across restarts)")
    result = jobs[job_id].copy()
    if result.get("fingerprint"):
        league = store.league(result["league_id"])
        f = result["fingerprint"]
        result["stale"] = f["draft_revision"]!=store.draft(league.id)["revision"] or f["catalog_revision"]!=store.catalog_meta(league.season)["revision"] or f["settings"]!=league.model_dump_json() or f["events"]!=[e.model_dump_json() for e in store.events()]
    return result


@app.post("/api/refresh/{season}")
def trigger_refresh(season:int):
    if not 2005<=season<=2100 or season==2099:
        raise ValueError("Invalid refresh season")
    return submit("refresh",lambda:refresh(season))


@app.get("/api/sources")
def sources():
    model_path=DATA/"model_2026.json"
    model=json.loads(model_path.read_text()) if model_path.exists() else {}
    model.pop("residuals",None)
    return {"model_selection":model,"sources":Cache().status(),"schema":__import__("app.data",fromlist=["supplemental_schema"]).supplemental_schema()}


@app.post("/api/leagues/{league_id}/backtest")
def run_backtest(league_id:str):
    league=store.league(league_id)
    return submit("backtest",lambda:backtest(historical(),league),league_id)


class PlayerImport(Strict):
    players:list[Player] = Field(max_length=3000)


@app.post("/api/players/{season}")
def import_players(season:int,body:PlayerImport):
    existing={p.id:p for p in store.players(season)}
    for p in body.players:
        if p.source=="independent":
            raise ValueError("Imported players need explicit source provenance")
        existing[p.id]=p
    from app.identity import Identity
    Identity(list(existing.values()))
    store.save_players(season,list(existing.values()))
    return {"players":len(existing)}


class MarketImport(Strict):
    rows:list[dict] = Field(max_length=3000)
    source:str
    as_of:str
    independent_errors:list[float] | None = None
    market_errors:list[float] | None = None
    provisional_weight:float = Field(default=0,ge=0,le=.5)


@app.post("/api/market/{season}")
def market(season:int,body:MarketImport):
    from datetime import datetime,timezone
    timestamp=datetime.fromisoformat(body.as_of.replace("Z","+00:00"))
    if timestamp.tzinfo is None or timestamp>datetime.now(timezone.utc):
        raise ValueError("Market timestamp must be timezone-aware and not in the future")
    players={p.id:p for p in store.players(season)}
    weight=ensemble_weight(body.independent_errors,body.market_errors) if body.independent_errors is not None and body.market_errors is not None else body.provisional_weight
    for row in body.rows:
        if row.get("id") not in players:
            raise ValueError("Unknown canonical market player ID")
        p=players[row["id"]]
        allowed={"adp","adp_sd","espn_rank","ecr","market_stats"}
        if set(row)-allowed-{"id"}:
            raise ValueError("Unknown market fields")
        merged=p.model_dump() | {k:v for k,v in row.items() if k in allowed} | {"market_weight":weight}
        p=Player.model_validate(merged)
        p.warnings=[w for w in p.warnings if w!="ADP unavailable"] if p.adp else p.warnings
        p.warnings.append(f"Market: {body.source} as of {body.as_of}; weight {weight:.2f}" + (" provisional" if body.independent_errors is None else " fitted to supplied errors"))
        players[p.id]=p
    store.save_players(season,list(players.values()))
    return {"updated":len(body.rows),"market_weight":weight}


@app.put("/api/leagues/{league_id}/rosters")
def roster_import(league_id:str,body:dict[str,list[str]]):
    from app.ownership import save_rosters
    return save_rosters(store,store.league(league_id),body)


@app.get("/api/news")
def news():
    from app.news import refresh_news
    return refresh_news()


@app.get("/api/events")
def events():
    return store.events()


@app.post("/api/events")
def event(body:Event):
    from datetime import datetime
    for value in (body.known_at,body.occurred_at):
        if datetime.fromisoformat(value.replace("Z","+00:00")).tzinfo is None:
            raise ValueError("Event timestamps require timezone")
    with store.connect() as c:
        seasons=[r[0] for r in c.execute("SELECT season FROM catalogs")]
    if not any(p.id==body.player_id for season in seasons for p in store.players(season)):
        raise ValueError("Unknown canonical event player ID")
    store.save_event(body)
    return body


class SeasonRequest(Strict):
    week:int=Field(default=1,ge=1,le=18)
    risk:str="balanced"
    faab_remaining:int=Field(default=100,ge=0)
    overrides:dict[str,float]=Field(default_factory=dict)


@app.post("/api/leagues/{league_id}/season")
def season(league_id:str,body:SeasonRequest):
    league,players,reports,picks,_=context(league_id)
    ids={p.player_id for p in picks if p.team==league.my_team}
    roster=[p for p in players if p.id in ids]
    return {"start_sit":start_sit(roster,reports,league,body.week,body.risk,body.overrides),"waivers":waivers(players,reports,league,picks,body.week,body.faab_remaining)}


class ESPNSyncRequest(Strict):
    league_id:str = Field(pattern=r"^[0-9]{1,20}$")
    my_team_id:int = Field(ge=0)
    season:int = Field(default=2026,ge=2005,le=2100)
    week:int = Field(default=1,ge=1,le=18)


@app.post("/api/espn/sync")
def sync_public(body:ESPNSyncRequest):
    from app.league_sync import fetch_public,save_snapshot
    return save_snapshot(store,fetch_public(body.league_id,body.season,body.week),body.my_team_id,body.season,body.week)


class ESPNImportRequest(ESPNSyncRequest):
    snapshot:dict


@app.post("/api/espn/import")
def import_espn(request_body:ESPNImportRequest):
    body=request_body.model_dump()
    from app.league_sync import save_snapshot
    config=ESPNSyncRequest.model_validate({k:body[k] for k in ('league_id','my_team_id','season','week')})
    if str(body.get('snapshot',{}).get('id'))!=config.league_id:
        raise ValueError('Snapshot and requested league differ')
    result=save_snapshot(store,body['snapshot'],config.my_team_id,config.season,config.week)
    save_connection(ConnectionRequest(url=f'https://fantasy.espn.com/football/team?leagueId={config.league_id}&teamId={config.my_team_id}&seasonId={config.season}',name=result['league']['team_names'][result['league']['my_team']]))
    return result


class LineupRequest(Strict):
    week:int = Field(default=1,ge=1,le=18)
    risk:str = 'balanced'
    refresh_injuries:bool = False


@app.post("/api/leagues/{league_id}/lineup")
def weekly_lineup(league_id:str,body:LineupRequest):
    from app.league_sync import sync_metadata
    from app.ownership import current_ownership
    from app.weekly import lineup_advice
    from app.injuries import injury_report
    league=store.league(league_id)
    players=apply_events(store.players(league.season),store.events(),now())
    draft=[Pick.model_validate(p) for p in store.draft(league_id)['picks']]
    owned,stamp=current_ownership(store,league,draft)
    ids=[p.player_id for p in owned if p.team==league.my_team]
    meta=sync_metadata(store,league_id)
    health=injury_report(force=body.refresh_injuries)
    if body.week>1:
        try:
            from app.data import BASE
            Cache().fetch(f'stats_{league.season}.parquet',f'{BASE}/stats_player/stats_player_week_{league.season}.parquet',ttl=3600)
        except Exception:
            pass  # weekly report explicitly marks missing recent usage
    from app.intelligence import weekly_context
    from app.tracking import blend_weight,save_players
    context_data=weekly_context(store,league,body.week)
    context_data['blend_weights']={pos:blend_weight(store,pos)[0] for pos in ('QB','RB','WR','TE','K','DST')}
    opponent=meta.get('opponents',{}).get(str(league.my_team))
    opponent_ids=[p.player_id for p in owned if opponent is not None and p.team==opponent]
    advice=lineup_advice(players,league,ids,meta,health,body.week,body.risk,context=context_data,opponent_ids=opponent_ids)
    save_players(store,league,advice)
    return advice



@app.get("/api/intelligence")
def intelligence_state():
    from app.intelligence import state
    return state(store)


@app.post("/api/intelligence/refresh")
def intelligence_refresh():
    from app.intelligence import refresh_intelligence
    if os.environ.get('FANTASY_DISABLE_AUTO_REFRESH')=='1':return {'id':'disabled','status':'complete','kind':'intelligence'}
    with job_lock:
        running=next((j.copy() for j in jobs.values() if j['kind']=='intelligence' and j['status'] in ('queued','running')),None)
    return running or submit('intelligence',lambda:refresh_intelligence(store))


from app.game_benchmarks import BenchmarkImport


@app.post('/api/intelligence/benchmarks')
def game_benchmark_import(body: BenchmarkImport):
    from app.game_benchmarks import import_benchmarks
    from app.intelligence import get_meta
    return import_benchmarks(store,body,get_meta(store,'intelligence-report',{}).get('games',[]))


@app.get('/api/intelligence/games/{game_id}/audit')
def game_audit_export(game_id: str):
    from app.game_learning import initialize,artifact
    initialize(store)
    with store.connect() as c:
        rows=c.execute('SELECT id,body,created_at FROM game_forecasts WHERE game_id=? ORDER BY id',(game_id,)).fetchall()
        reviews=c.execute('SELECT body,created_at FROM game_result_versions WHERE game_id=? ORDER BY id',(game_id,)).fetchall()
    if not rows:raise HTTPException(404,'No archived forecast for this game')
    forecasts=[json.loads(r['body'])|{'forecast_id':r['id'],'saved_at':r['created_at']} for r in rows]
    ids={f.get('base_artifact') for f in forecasts}
    for f in forecasts:
        ids.add(f.get('learning',{}).get('active_artifact'))
        ids.add((f.get('learning',{}).get('candidate') or {}).get('artifact'))
    return {'game_id':game_id,'forecasts':forecasts,'reviews':[json.loads(r['body'])|{'reviewed_at':r['created_at']} for r in reviews],
            'artifacts':{key:artifact(store,key) for key in ids if key},'note':'Local pregame snapshots and model artifacts. Result revisions do not rewrite forecasts.'}


class TradeRequest(Strict):
    give:list[str]
    receive:list[str]
    week:int=Field(default=1,ge=1,le=18)


@app.post("/api/leagues/{league_id}/trade")
def trade_route(league_id:str,body:TradeRequest):
    league,players,reports,picks,_=context(league_id)
    return trade(players,reports,league,picks,body.give,body.receive,body.week)


@app.get("/api/leagues/{league_id}/export")
def export(league_id:str):
    league=store.league(league_id)
    return {"league":league,"draft":store.draft(league_id),"players":store.players(league.season),"events":store.events()}


dist=ROOT / "frontend" / "dist"
if (dist / "assets").exists():
    app.mount("/assets",StaticFiles(directory=dist/"assets"),name="assets")


@app.get("/")
def root():
    if (dist / "index.html").exists():
        return FileResponse(dist/"index.html")
    return JSONResponse({"message":"Build frontend: cd frontend && npm install && npm run build. API is ready at /docs."})
