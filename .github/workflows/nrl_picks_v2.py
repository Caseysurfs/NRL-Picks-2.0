# nrl_picks_v2.py — NRL Picks 2.0 (GitHub Actions)

import os, math, datetime as dt
from pathlib import Path
import pandas as pd, numpy as np, pytz, requests

API_BASE    = "https://api.the-odds-api.com/v4"
SPORT_KEY   = "rugbyleague_nrl"
REGIONS     = os.getenv("REGIONS", "au")
MARKETS     = "h2h"
ODDS_FORMAT = "decimal"

BANKROLL     = float(os.getenv("BANKROLL", "1000"))
KELLY_FACTOR = float(os.getenv("KELLY_FACTOR", "0.5"))
EDGE_MIN     = float(os.getenv("EDGE_MIN", "0.02"))
MAX_KELLY    = float(os.getenv("MAX_KELLY", "0.02"))

TZ_AEST = pytz.timezone("Australia/Brisbane")
API_KEY = os.getenv("THE_ODDS_API_KEY")

def now_aest_str(): return dt.datetime.now(TZ_AEST).strftime("%Y-%m-%d %H:%M")
def to_aest(iso_ts): return dt.datetime.fromisoformat(iso_ts.replace("Z","+00:00")).astimezone(TZ_AEST)
def logistic(x): return 1.0/(1.0+10.0**(-x/400.0))

def devig_two_way(a,b):
    if not all(isinstance(x,(int,float)) for x in (a,b)): return (None,None)
    if a<=1 or b<=1: return (None,None)
    pa,pb=1/a,1/b; s=pa+pb; return pa/s,pb/s

def kelly_fraction(p,odds,k):
    if odds<=1: return 0.0
    b=odds-1.0
    f=(p*(b+1.0)-1.0)/b
    f=max(0.0,k*f)
    return min(f, MAX_KELLY)

def fetch_scores_fallback(api_key):
    windows=[365,180,120,90,60,30,14,7,3]
    last=""
    for d in windows:
        try:
            r=requests.get(f"{API_BASE}/sports/{SPORT_KEY}/scores",
                           params={"daysFrom":d,"apiKey":api_key}, timeout=30)
            if r.status_code!=200:
                last=f"{r.status_code} {r.text[:120]}"; continue
            rows=[]
            for g in r.json():
                if not g.get("completed"): continue
                home,away=g.get("home_team"),g.get("away_team")
                if not home or not away: continue
                pm={s["name"]:float(s["score"]) for s in g.get("scores",[])}
                if home in pm and away in pm:
                    t=dt.datetime.fromisoformat(g["commence_time"].replace("Z","+00:00"))
                    rows.append({"date":t,"home":home,"away":away,"hs":pm[home],"as":pm[away]})
            if rows:
                print(f"[scores] {len(rows)} completed games using daysFrom={d}.")
                return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        except Exception as e:
            last=str(e); continue
    print(f"[scores] No historical scores found. Last error: {last}")
    return pd.DataFrame(columns=["date","home","away","hs","as"])

def fetch_odds(api_key):
    r=requests.get(f"{API_BASE}/sports/{SPORT_KEY}/odds",
                   params={"apiKey":api_key,"regions":REGIONS,"markets":MARKETS,"oddsFormat":ODDS_FORMAT},
                   timeout=30)
    r.raise_for_status(); return r.json()

def run_elo(df, K=30.0, HFA=60.0):
    ratings={}; eps=1e-9; ll=0.0; n=0
    for _,r in df.iterrows():
        rh,ra=ratings.get(r.home,1500.0),ratings.get(r.away,1500.0)
        p_home=logistic((rh-ra)+HFA)
        if r.hs==r.as: y=0.5
        elif r.hs>r.as: y=1.0
        else: y=0.0
        margin=abs(r.hs-r.as); k_eff=K*(1.0+math.log1p(margin))
        ratings[r.home]=rh+k_eff*(y-p_home); ratings[r.away]=ra-k_eff*(y-p_home)
        ll-= y*math.log(max(p_home,eps)) + (1-y)*math.log(max(1-p_home,eps)); n+=1
    return ratings, ll/max(n,1)

def tune_elo(df):
    best=None; params={"K":30.0,"HFA":60.0}
    for K in [15,25,35]:
        for HFA in [40,60,80]:
            _,loss=run_elo(df,K,HFA)
            if best is None or loss<best: best=loss; params={"K":K,"HFA":HFA}
    return params

def main():
    if not API_KEY: raise SystemExit("Missing THE_ODDS_API_KEY secret/env var.")

    hist=fetch_scores_fallback(API_KEY)
    if hist.empty:
        print("[elo] No history; using flat ratings, HFA=60."); ratings={}; params={"HFA":60}
    else:
        params=tune_elo(hist); ratings,_=run_elo(hist, params["K"], params["HFA"])
        print(f"[elo] Trained on {len(hist)} matches. Best: K={params['K']}, HFA={params['HFA']}.")

    rows=[]
    try: events=fetch_odds(API_KEY)
    except Exception as e:
        print(f"[odds] Error: {e}"); events=[]

    for ev in events:
        home,away=ev.get("home_team"),ev.get("away_team")
        if not home or not away: continue
        ko=to_aest(ev["commence_time"])

        best_home=best_away=None; book_h=book_a=None
        for bk in ev.get("bookmakers",[]):
            bkey=bk.get("key")
            for m in bk.get("markets",[]):
                if m.get("key")!="h2h": continue
                d={o.get("name"):o.get("price") for o in m.get("outcomes",[])}
                ph,pa=d.get(home),d.get(away)
                if isinstance(ph,(int,float)) and (best_home is None or ph>best_home): best_home,book_h=ph,bkey
                if isinstance(pa,(int,float)) and (best_away is None or pa>best_away): best_away,book_a=pa,bkey
        if best_home is None or best_away is None: continue

        m_home,m_away=devig_two_way(best_home,best_away)
        if m_home is None: continue

        rh,ra=ratings.get(home,1500.0),ratings.get(away,1500.0)
        p_home=logistic((rh-ra)+params["HFA"]); p_away=1.0-p_home
        edge_home,edge_away=p_home-m_home, p_away-m_away

        if edge_home>=edge_away:
            side,p,price,mp,edge,book,sel="HOME",p_home,best_home,m_home,edge_home,book_h,home
        else:
            side,p,price,mp,edge,book,sel="AWAY",p_away,best_away,m_away,edge_away,book_a,away

        if edge<EDGE_MIN: continue

        stake_frac=kelly_fraction(p,price,KELLY_FACTOR)
        stake_amt=round(BANKROLL*stake_frac,2)

        rows.append({
            "Generated (AEST)": now_aest_str(),
            "Kickoff (AEST)": ko.strftime("%Y-%m-%d %H:%M"),
            "Match": f"{home} vs {away}",
            "Pick": f"{side} — {sel}",
            "Book": book,
            "Best Odds": price,
            "Market p (de-vig)": round(mp,4),
            "Model p (Elo)": round(float(p),4),
            "Edge": round(float(edge),4),
            "Kelly % (½-Kelly, cap)": round(stake_frac*100,2),
            f"Stake (Bankroll={int(BANKROLL)})": stake_amt,
            "Potential Profit (if win)": round(stake_amt*(price-1),2),
            "Potential Return (if win)": round(stake_amt*price,2),
            "EV Profit (model)": round(stake_amt*(p*(price-1)-(1-p)),2),
        })

    df=pd.DataFrame(rows).sort_values(["Kickoff (AEST)","Edge"], ascending=[True,False]).reset_index(drop=True)
    outdir=Path("reports"); outdir.mkdir(parents=True, exist_ok=True)
    fname=outdir / f"nrl_picks_{dt.datetime.now(TZ_AEST).strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(fname, index=False)
    print(f"[output] Saved {len(df)} picks -> {fname}")
    if df.empty:
        print("[note] Empty CSV = no edges >= EDGE_MIN or odds not available for your plan/regions.")
    print("[responsible] Bet sensibly. BetStop & Gambling Help Online 1800 858 858 (AU).")

if __name__=="__main__":
    main()
