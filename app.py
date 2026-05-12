"""
SmartParking AI — Edge V2X Prototype
=====================================
Thesis: Optimisation de la gestion dynamique des places de stationnement urbain
        à l'aide de l'Intelligence Artificielle embarquée en Edge dans un environnement V2X

Run:
    pip install fastapi uvicorn
    python app.py
Then open http://localhost:8000
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import math, random, time, threading, json
from pathlib import Path

app = FastAPI(title="SmartParking AI — Edge V2X")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
Path("static").mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─────────────────────────────────────────────────────────────────────────────
# STATIC DATA  (from config + XML files of the real project)
# ─────────────────────────────────────────────────────────────────────────────
AGENTS = {
    "agent_1": {"name":"Zone Nord-Est","lat":49.635,"lon":6.170,"color":"#1F3864","radius_m":1200,
                "parkings":["P1","P2","P3","P19","P20","P21","P22"],
                "capacity":540,"traffic_score":14367,"centroid_x":8776,"centroid_y":8428},
    "agent_2": {"name":"Zone Centre",  "lat":49.619,"lon":6.133,"color":"#2E74B5","radius_m":1500,
                "parkings":["P4","P5","P12","P15","P23","P24","P28"],
                "capacity":500,"traffic_score":12783,"centroid_x":5848,"centroid_y":6352},
    "agent_3": {"name":"Zone Sud",     "lat":49.607,"lon":6.142,"color":"#0070C0","radius_m":2200,
                "parkings":["P6","P7","P8","P10","P11","P27"],
                "capacity":480,"traffic_score":3624,"centroid_x":6604,"centroid_y":4779},
    "agent_4": {"name":"Zone Ouest",   "lat":49.625,"lon":6.106,"color":"#375623","radius_m":1750,
                "parkings":["P9","P13","P14","P16","P17","P18","P25","P26"],
                "capacity":480,"traffic_score":1154,"centroid_x":3684,"centroid_y":7089},
}

PARKINGS = {
    "P1":  {"lat":49.638,"lon":6.162,"cap":80, "agent":"agent_1","name":"P1 – Kirchberg"},
    "P2":  {"lat":49.636,"lon":6.168,"cap":80, "agent":"agent_1","name":"P2 – Kiem"},
    "P3":  {"lat":49.632,"lon":6.174,"cap":60, "agent":"agent_1","name":"P3 – Cents"},
    "P4":  {"lat":49.622,"lon":6.128,"cap":60, "agent":"agent_2","name":"P4 – Gare Nord"},
    "P5":  {"lat":49.617,"lon":6.135,"cap":100,"agent":"agent_2","name":"P5 – Bonnevoie"},
    "P6":  {"lat":49.610,"lon":6.138,"cap":100,"agent":"agent_3","name":"P6 – Hollerich"},
    "P7":  {"lat":49.605,"lon":6.145,"cap":80, "agent":"agent_3","name":"P7 – Gasperich"},
    "P8":  {"lat":49.602,"lon":6.150,"cap":80, "agent":"agent_3","name":"P8 – Cloche d'Or"},
    "P9":  {"lat":49.628,"lon":6.100,"cap":60, "agent":"agent_4","name":"P9 – Bertrange"},
    "P10": {"lat":49.612,"lon":6.135,"cap":60, "agent":"agent_3","name":"P10 – Cessange"},
    "P11": {"lat":49.608,"lon":6.148,"cap":60, "agent":"agent_3","name":"P11 – Merl"},
    "P12": {"lat":49.624,"lon":6.138,"cap":60, "agent":"agent_2","name":"P12 – Gare CFL"},
    "P13": {"lat":49.622,"lon":6.108,"cap":60, "agent":"agent_4","name":"P13 – Strassen"},
    "P14": {"lat":49.618,"lon":6.102,"cap":60, "agent":"agent_4","name":"P14 – Mamer"},
    "P15": {"lat":49.615,"lon":6.130,"cap":60, "agent":"agent_2","name":"P15 – Belair"},
    "P16": {"lat":49.632,"lon":6.112,"cap":60, "agent":"agent_4","name":"P16 – Leudelange"},
    "P17": {"lat":49.625,"lon":6.095,"cap":60, "agent":"agent_4","name":"P17 – Koerich"},
    "P18": {"lat":49.620,"lon":6.115,"cap":60, "agent":"agent_4","name":"P18 – Capellen"},
    "P19": {"lat":49.640,"lon":6.175,"cap":80, "agent":"agent_1","name":"P19 – Hamm"},
    "P20": {"lat":49.643,"lon":6.165,"cap":80, "agent":"agent_1","name":"P20 – Weimerskirch"},
    "P21": {"lat":49.633,"lon":6.160,"cap":80, "agent":"agent_1","name":"P21 – Pulvermühl"},
    "P22": {"lat":49.630,"lon":6.172,"cap":80, "agent":"agent_1","name":"P22 – Neudorf"},
    "P23": {"lat":49.619,"lon":6.142,"cap":80, "agent":"agent_2","name":"P23 – Centre-Ville"},
    "P24": {"lat":49.626,"lon":6.125,"cap":60, "agent":"agent_2","name":"P24 – Limpertsberg"},
    "P25": {"lat":49.635,"lon":6.098,"cap":60, "agent":"agent_4","name":"P25 – Dommeldange"},
    "P26": {"lat":49.615,"lon":6.118,"cap":60, "agent":"agent_4","name":"P26 – Pétange"},
    "P27": {"lat":49.600,"lon":6.142,"cap":100,"agent":"agent_3","name":"P27 – Esch Route"},
    "P28": {"lat":49.612,"lon":6.140,"cap":80, "agent":"agent_2","name":"P28 – Bonnevoie Sud"},
}

FL_HISTORY = {
    "rounds": [
        {"round":1,"epsilon":0.900,
         "fit":  {"reward":4497.63,"assigned":568.5,"loss":0.2189,"buffer":573.5,"steps":573.5},
         "eval": {"reward":1471.90,"assigned":145.25}},
        {"round":2,"epsilon":0.720,
         "fit":  {"reward":4572.64,"assigned":577.5,"loss":0.1665,"buffer":1154.5,"steps":581.0},
         "eval": {"reward":1464.30,"assigned":143.50}},
    ],
    "per_agent": {
        "agent_1":{
            "fit": [{"round":1,"reward":2861.03,"assigned":289,"loss":0.2806,"steps":294},
                    {"round":2,"reward":2866.38,"assigned":300,"loss":0.2003,"steps":308}],
            "eval":[{"round":1,"reward":791.94,"assigned":81},
                    {"round":2,"reward":703.85,"assigned":71}]},
        "agent_2":{
            "fit": [{"round":1,"reward":5276.28,"assigned":767,"loss":0.1735,"steps":769},
                    {"round":2,"reward":5151.99,"assigned":759,"loss":0.1496,"steps":760}],
            "eval":[{"round":1,"reward":1970.09,"assigned":198},
                    {"round":2,"reward":2030.67,"assigned":199}]},
        "agent_3":{
            "fit": [{"round":1,"reward":6374.95,"assigned":820,"loss":0.1804,"steps":826},
                    {"round":2,"reward":6585.19,"assigned":836,"loss":0.1465,"steps":838}],
            "eval":[{"round":1,"reward":2060.72,"assigned":198},
                    {"round":2,"reward":2019.39,"assigned":198}]},
        "agent_4":{
            "fit": [{"round":1,"reward":3478.25,"assigned":398,"loss":0.2413,"steps":405},
                    {"round":2,"reward":3686.99,"assigned":415,"loss":0.1695,"steps":418}],
            "eval":[{"round":1,"reward":1064.86,"assigned":104},
                    {"round":2,"reward":1103.30,"assigned":106}]},
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# LIVE SIMULATION STATE
# ─────────────────────────────────────────────────────────────────────────────
class SimState:
    def __init__(self):
        random.seed(42)
        self.step = 0
        self.occupancy  = {pid: random.randint(3, int(info["cap"]*0.45)) for pid,info in PARKINGS.items()}
        self.incoming   = {pid: random.randint(0,4)  for pid in PARKINGS}
        self.traffic_pressure = 1.0
        self.total_assigned   = 0
        self.recent_decisions = []          # last 20 assignments
        self.reward_history   = []          # rolling reward per step
        self.lock = threading.Lock()

    def free(self, pid):
        return max(0, PARKINGS[pid]["cap"] - self.occupancy[pid] - self.incoming[pid])

    def occ_ratio(self, pid):
        return self.occupancy[pid] / PARKINGS[pid]["cap"]

    def pred_ratio(self, pid):
        cap = PARKINGS[pid]["cap"]
        return min((self.occupancy[pid] + min(self.incoming[pid], int(0.35*cap))) / cap, 1.0)

    def tick(self):
        with self.lock:
            self.step += 1
            cycle = self.step % 1200
            if   200 <= cycle <= 450:  self.traffic_pressure = 1.30 + random.uniform(0,.10)
            elif 700 <= cycle <= 950:  self.traffic_pressure = 1.38 + random.uniform(0,.12)
            else:                      self.traffic_pressure = 1.00 + random.uniform(0,.15)

            for pid, info in PARKINGS.items():
                cap = info["cap"]
                occ = self.occupancy[pid]
                if occ > 0 and random.random() < 0.10:
                    self.occupancy[pid] = max(0, occ - random.randint(1,3))
                if self.free(pid) > 1 and random.random() < 0.13:
                    self.occupancy[pid] = min(cap, self.occupancy[pid] + random.randint(1,4))
                self.incoming[pid] = max(0, random.randint(0, max(1, self.free(pid)//3)))

sim = SimState()

def _bg():
    while True:
        sim.tick()
        time.sleep(2)

threading.Thread(target=_bg, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# SCORING  (replicates DQN candidate scoring from multi_agent_env.py)
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1,lon1,lat2,lon2):
    R=6371000
    f1,f2=math.radians(lat1),math.radians(lat2)
    df=math.radians(lat2-lat1); dl=math.radians(lon2-lon1)
    a=math.sin(df/2)**2+math.cos(f1)*math.cos(f2)*math.sin(dl/2)**2
    return R*2*math.atan2(math.sqrt(a),math.sqrt(1-a))

def dyn_price(pid, mode, tp):
    cap=PARKINGS[pid]["cap"]
    occ_r=sim.occupancy[pid]/cap
    pred_r=sim.pred_ratio(pid)
    p=2.0+0.80*occ_r+0.60*pred_r
    p+={"cheap":-0.30,"close":+0.25}.get(mode,+0.10)
    p*=min(max(tp,0.9),1.3)
    if pred_r>=.90: p+=1.50
    elif pred_r>=.75: p+=0.90
    elif pred_r>=.60: p+=0.45
    return round(max(1.5,min(p,7.0)),2)

def candidate_score(pid, dist, mode, tp):
    price=dyn_price(pid,mode,tp)
    cap=PARKINGS[pid]["cap"]
    free=sim.free(pid)
    pred_r=sim.pred_ratio(pid)
    incoming=sim.incoming[pid]
    dn=min(dist/3000,1.0); pn=min(price/6.5,1.0)
    ir=min(incoming/cap,1.0); fr=free/cap
    if mode=="close":   s=1.65*dn+0.10*pn+0.18*pred_r+0.05*ir-0.08*fr
    elif mode=="cheap": s=0.45*dn+1.30*pn+0.18*pred_r+0.05*ir-0.08*fr
    else:               s=0.75*dn+0.75*pn+0.22*pred_r+0.05*ir-0.10*fr
    if pred_r>0.60: s+=0.25
    if pred_r>0.80: s+=0.70
    if pred_r>0.92: s+=1.60
    return s, price

# ─────────────────────────────────────────────────────────────────────────────
# API ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root(): return FileResponse("static/index.html")

@app.get("/api/config")
def config():
    return JSONResponse({"agents": AGENTS, "parkings": {
        pid: {**info, "id": pid} for pid, info in PARKINGS.items()
    }})

@app.get("/api/state")
def state():
    with sim.lock:
        pkgs={}
        for pid,info in PARKINGS.items():
            occ=sim.occupancy[pid]; cap=info["cap"]
            r=occ/cap
            pkgs[pid]={
                "occupancy":occ,"capacity":cap,
                "free":sim.free(pid),"incoming":sim.incoming[pid],
                "occ_ratio":round(r,3),
                "pred_ratio":round(sim.pred_ratio(pid),3),
                "price_balanced":dyn_price(pid,"balanced",sim.traffic_pressure),
                "status":"full" if r>=.98 else "high" if r>=.75 else "medium" if r>=.40 else "low",
            }
        total_cap=sum(PARKINGS[p]["cap"] for p in PARKINGS)
        total_occ=sum(sim.occupancy[p] for p in PARKINGS)
        return JSONResponse({
            "step":sim.step,"traffic_pressure":round(sim.traffic_pressure,3),
            "total_assigned":sim.total_assigned,
            "global_occ_ratio":round(total_occ/total_cap,3),
            "total_free":total_cap-total_occ,
            "parkings":pkgs,
            "recent_decisions":sim.recent_decisions[-10:],
        })

class ReqBody(BaseModel):
    lat:float; lon:float; mode:str="balanced"

@app.post("/api/recommend")
def recommend(b: ReqBody):
    mode=b.mode if b.mode in ["close","cheap","balanced"] else "balanced"
    tp=sim.traffic_pressure
    results=[]
    for pid,info in PARKINGS.items():
        if sim.free(pid)<=0: continue
        if sim.pred_ratio(pid)>=0.99: continue
        dist=haversine(b.lat,b.lon,info["lat"],info["lon"])
        if dist>4000: continue
        sc,price=candidate_score(pid,dist,mode,tp)
        results.append({
            "id":pid,"name":info["name"],"agent":info["agent"],
            "lat":info["lat"],"lon":info["lon"],
            "distance_m":int(dist),"price":price,
            "free_slots":sim.free(pid),"capacity":info["cap"],
            "occ_ratio":round(sim.occ_ratio(pid),3),
            "pred_ratio":round(sim.pred_ratio(pid),3),
            "score":round(sc,4),"rank":0,
        })
    results.sort(key=lambda x:x["score"])
    top5=results[:5]
    for i,r in enumerate(top5): r["rank"]=i+1

    # nearest responsible agent
    best_agent=min(AGENTS.items(),key=lambda a:haversine(b.lat,b.lon,a[1]["lat"],a[1]["lon"]))[0]

    with sim.lock:
        sim.total_assigned+=1
        decision={
            "step":sim.step,"mode":mode,
            "parking":top5[0]["id"] if top5 else None,
            "agent":top5[0]["agent"] if top5 else best_agent,
            "reward":round(10*(1-top5[0]["occ_ratio"]),2) if top5 else 0,
            "distance":top5[0]["distance_m"] if top5 else 0,
        }
        sim.recent_decisions.append(decision)
        if len(sim.recent_decisions)>50: sim.recent_decisions.pop(0)

    return JSONResponse({
        "mode":mode,"agent":best_agent,
        "recommendation":top5[0] if top5 else None,
        "top5":top5,"step":sim.step,
        "traffic_pressure":round(tp,3),
    })

@app.get("/api/fl")
def fl_data(): return JSONResponse(FL_HISTORY)

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    print("\n" + "="*55)
    print("  SmartParking AI — Edge V2X Prototype")
    print(f"  Ouvrez votre navigateur sur : http://localhost:{port}")
    print("="*55 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port)

