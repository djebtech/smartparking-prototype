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

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import math, random, time, threading, uuid
from datetime import datetime, timezone
from pathlib import Path

app = FastAPI(title="MADINA — Edge V2X")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Use absolute path so Railway / any deployment finds the static folder correctly
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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
# MODE RADIUS LIMITS  (used by /api/recommend + exposed in /api/config)
# ─────────────────────────────────────────────────────────────────────────────
MODE_RADIUS = {
    "close":    {"normal": 500,  "dense": 700},
    "cheap":    {"normal": 1500, "dense": 2000},
    "balanced": {"normal": 1000, "dense": 1400},
}

# ─────────────────────────────────────────────────────────────────────────────
# RESERVATIONS & RL FEEDBACK  (Module 4)
# ─────────────────────────────────────────────────────────────────────────────
RESERVATIONS: dict = {}      # reservation_id → reservation dict
RL_FEEDBACK_LOG: list = []   # feedback entries injected into RL memory

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL MODEL METADATA  (Module 3 — simulated Flower weights)
# ─────────────────────────────────────────────────────────────────────────────
GLOBAL_MODEL = {
    "version": 2,
    "round": 2,
    "epsilon": 0.720,
    "architecture": "DuelingDQN",
    "state_dim": 48,
    "action_dim": 5,
    "updated_at": "2025-12-01T12:00:00Z",
    "aggregation": "FedAvg",
    "num_clients": 4,
    "layers": [
        {"name": "fc1",     "shape": [256, 48],  "norm": 1.4821},
        {"name": "fc2",     "shape": [128, 256], "norm": 1.1203},
        {"name": "adv1",    "shape": [64, 128],  "norm": 0.9347},
        {"name": "adv_out", "shape": [5, 64],    "norm": 0.6218},
        {"name": "val1",    "shape": [64, 128],  "norm": 0.8932},
        {"name": "val_out", "shape": [1, 64],    "norm": 0.4105},
    ],
    "total_params": 256*48 + 128*256 + 64*128 + 5*64 + 64*128 + 1*64,
}

# ─────────────────────────────────────────────────────────────────────────────
# LIVE SIMULATION STATE
# ─────────────────────────────────────────────────────────────────────────────
class SimState:
    def __init__(self):
        random.seed(42)
        self.step = 0
        # FIX 1 — Seed initial occupancy proportional to zone traffic_score
        # agent_1 (traffic 14367) → 55–75%, agent_2 (12783) → 45–65%,
        # agent_3 (3624) → 28–48%, agent_4 (1154) → 12–28%
        _occ_bands = {
            "agent_1": (0.55, 0.75),
            "agent_2": (0.45, 0.65),
            "agent_3": (0.28, 0.48),
            "agent_4": (0.12, 0.28),
        }
        self.occupancy = {}
        for pid, info in PARKINGS.items():
            lo, hi = _occ_bands[info["agent"]]
            self.occupancy[pid] = random.randint(int(info["cap"] * lo), int(info["cap"] * hi))
        self.incoming   = {pid: random.randint(0,4)  for pid in PARKINGS}
        self.traffic_pressure = 1.0
        self.total_assigned   = 0
        self.recent_decisions = []
        self.reward_history   = []
        self.lock = threading.Lock()
        # FIX 2 — Per-vehicle dynamic stay duration
        # vehicle_durations[pid] = list of (arrival_step, stay_steps)
        # At 2 s/tick: 450 ticks ≈ 15 min | 1 800 ≈ 60 min | 3 600 ≈ 120 min
        self.vehicle_durations: dict = {}
        for pid in PARKINGS:
            occ = self.occupancy[pid]
            # Seed pre-existing vehicles: random past arrival + random stay length
            self.vehicle_durations[pid] = [
                (-random.randint(0, 500), int(max(450, random.gauss(1800, 600))))
                for _ in range(occ)
            ]

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
            if   200 <= cycle <= 450:  self.traffic_pressure = 1.30 + random.uniform(0, .10)
            elif 700 <= cycle <= 950:  self.traffic_pressure = 1.38 + random.uniform(0, .12)
            else:                      self.traffic_pressure = 1.00 + random.uniform(0, .15)

            # Per-agent drift targets aligned with real traffic_score ranking
            _drift = {
                "agent_1": {"target": 0.65, "prob": 0.32, "inc": (1, 4)},  # busiest zone
                "agent_2": {"target": 0.55, "prob": 0.24, "inc": (1, 3)},
                "agent_3": {"target": 0.38, "prob": 0.16, "inc": (1, 3)},
                "agent_4": {"target": 0.20, "prob": 0.10, "inc": (1, 2)},  # quietest zone
            }

            for pid, info in PARKINGS.items():
                cap = info["cap"]
                ag  = info["agent"]

                # ── FIX 2a: Duration-based departures ──────────────────────
                remaining = []
                departed  = 0
                for (arr, stay) in self.vehicle_durations[pid]:
                    if self.step >= arr + stay:
                        departed += 1
                    else:
                        remaining.append((arr, stay))
                self.vehicle_durations[pid] = remaining
                if departed:
                    self.occupancy[pid] = max(0, self.occupancy[pid] - departed)

                # ── FIX 2b: Arrivals with individual random stay duration ──
                if self.free(pid) > 1 and random.random() < 0.13:
                    n = min(random.randint(1, 4), self.free(pid))
                    self.occupancy[pid] = min(cap, self.occupancy[pid] + n)
                    for _ in range(n):
                        # Each vehicle gets its own stay length: ~15–120 min
                        stay_ticks = int(max(450, random.gauss(1800, 600)))
                        self.vehicle_durations[pid].append((self.step, stay_ticks))

                # ── FIX 1: Drift toward traffic-proportional occupancy ─────
                d      = _drift[ag]
                target = int(cap * d["target"])
                if self.occupancy[pid] < target and random.random() < d["prob"]:
                    self.occupancy[pid] = min(target, self.occupancy[pid] + random.randint(*d["inc"]))
                elif self.occupancy[pid] > target + int(cap * 0.15) and random.random() < 0.08:
                    self.occupancy[pid] = max(target, self.occupancy[pid] - random.randint(1, 2))

                self.incoming[pid] = max(0, random.randint(0, max(1, self.free(pid)//3)))

sim = SimState()

def _bg():
    while True:
        sim.tick()
        time.sleep(2)

threading.Thread(target=_bg, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    f1, f2 = math.radians(lat1), math.radians(lat2)
    df = math.radians(lat2-lat1); dl = math.radians(lon2-lon1)
    a = math.sin(df/2)**2 + math.cos(f1)*math.cos(f2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def _kmeans(points, k, max_iter=15):
    """Lightweight K-Means on (lat, lon) — no sklearn dependency."""
    if len(points) <= k:
        return list(range(len(points)))
    rng = random.Random(42)
    centroids = [points[i] for i in rng.sample(range(len(points)), k)]
    labels = [0] * len(points)
    for _ in range(max_iter):
        for i, p in enumerate(points):
            labels[i] = min(range(k), key=lambda c: (p[0]-centroids[c][0])**2 + (p[1]-centroids[c][1])**2)
        new_c = []
        for c in range(k):
            pts = [points[i] for i in range(len(points)) if labels[i] == c]
            new_c.append(
                (sum(p[0] for p in pts)/len(pts), sum(p[1] for p in pts)/len(pts)) if pts else centroids[c]
            )
        if new_c == centroids:
            break
        centroids = new_c
    return labels

# ─────────────────────────────────────────────────────────────────────────────
# SCORING  (replicates DQN candidate scoring from multi_agent_env.py)
# ─────────────────────────────────────────────────────────────────────────────
def dyn_price(pid, mode, tp):
    cap    = PARKINGS[pid]["cap"]
    occ_r  = sim.occupancy[pid] / cap
    pred_r = sim.pred_ratio(pid)
    p = 2.0 + 0.80 * occ_r + 0.60 * pred_r
    p += {"cheap": -0.30, "close": +0.25}.get(mode, +0.10)
    p *= min(max(tp, 0.9), 1.3)
    if pred_r >= .90: p += 1.50
    elif pred_r >= .75: p += 0.90
    elif pred_r >= .60: p += 0.45
    # MOD 3 — Hyper-Tarification: dissuade les profils "cheap" des zones denses
    if mode == "cheap" and pred_r >= 0.65:
        p += 1.20 * min((pred_r - 0.65) / 0.35, 1.0)
        if pred_r >= 0.80:
            p += 0.80
    return round(max(1.5, min(p, 7.0)), 2)

def candidate_score(pid, dist, mode, tp):
    price   = dyn_price(pid, mode, tp)
    cap     = PARKINGS[pid]["cap"]
    free    = sim.free(pid)
    pred_r  = sim.pred_ratio(pid)
    incoming = sim.incoming[pid]
    dn = min(dist / 3000, 1.0)
    pn = min(price / 6.5,  1.0)
    ir = min(incoming / cap, 1.0)
    fr = free / cap
    if mode == "close":   s = 1.65*dn + 0.10*pn + 0.18*pred_r + 0.05*ir - 0.08*fr
    elif mode == "cheap": s = 0.45*dn + 1.30*pn + 0.18*pred_r + 0.05*ir - 0.08*fr
    else:                 s = 0.75*dn + 0.75*pn + 0.22*pred_r + 0.05*ir - 0.10*fr
    if pred_r > 0.60: s += 0.25
    if pred_r > 0.80: s += 0.70
    if pred_r > 0.92: s += 1.60
    # MOD 1 — Malus de Distance: pénalité stricte au-delà de 1 km en urbain
    if dist > 1000:
        s += 0.40 * ((dist - 1000) / 1000)
    # MOD 2 — Bonus de Repeuplement: attirer les véhicules vers la zone sous-occupée d'Agent 1
    if PARKINGS[pid]["agent"] == "agent_1" and fr >= 0.60:
        s -= 0.25
    return s, price

# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────
class ReqBody(BaseModel):
    lat: float; lon: float; mode: str = "balanced"

class SearchBody(BaseModel):
    lat: float
    lon: float
    rayon_recherche: float = 500.0   # meters
    prix_max: float = 5.0            # €/h  (range 3–7)
    mode: str = "balanced"

class ReservationBody(BaseModel):
    parking_id: str
    lat: float
    lon: float
    mode: str = "balanced"

class FeedbackBody(BaseModel):
    reservation_id: str
    action: str       # "ACCEPTER" | "REFUSER"
    rating: int       # 1–5

class LocalUpdateBody(BaseModel):
    agent_id: str
    round: int
    layer_norms: List[float]

# ─────────────────────────────────────────────────────────────────────────────
# LEGACY API ROUTES  (unchanged — keeps existing frontend working)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/login")
def login_page(): return FileResponse(str(STATIC_DIR / "login.html"))

@app.get("/")
def root(): return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/api/config")
def config():
    return JSONResponse({"agents": AGENTS, "parkings": {
        pid: {**info, "id": pid} for pid, info in PARKINGS.items()
    }, "mode_radius": MODE_RADIUS})

@app.get("/api/state")
def state_endpoint():
    with sim.lock:
        pkgs = {}
        for pid, info in PARKINGS.items():
            occ = sim.occupancy[pid]; cap = info["cap"]
            r = occ / cap
            pkgs[pid] = {
                "occupancy": occ, "capacity": cap,
                "free": sim.free(pid), "incoming": sim.incoming[pid],
                "occ_ratio": round(r, 3),
                "pred_ratio": round(sim.pred_ratio(pid), 3),
                "price_balanced": dyn_price(pid, "balanced", sim.traffic_pressure),
                "status": "high" if r>=.80 else "medium" if r>=.50 else "low",
            }
        total_cap = sum(PARKINGS[p]["cap"] for p in PARKINGS)
        total_occ = sum(sim.occupancy[p] for p in PARKINGS)
        return JSONResponse({
            "step": sim.step, "traffic_pressure": round(sim.traffic_pressure, 3),
            "total_assigned": sim.total_assigned,
            "global_occ_ratio": round(total_occ/total_cap, 3),
            "total_free": total_cap - total_occ,
            "parkings": pkgs,
            "recent_decisions": sim.recent_decisions[-10:],
        })

@app.post("/api/recommend")
def recommend(b: ReqBody):
    mode = b.mode if b.mode in ["close","cheap","balanced"] else "balanced"
    tp = sim.traffic_pressure
    pressure_key = "dense" if tp >= 1.3 else "normal"
    max_dist = MODE_RADIUS[mode][pressure_key]
    results = []
    for pid, info in PARKINGS.items():
        if sim.free(pid) <= 0: continue
        if sim.pred_ratio(pid) >= 0.99: continue
        dist = haversine(b.lat, b.lon, info["lat"], info["lon"])
        if dist > max_dist: continue
        sc, price = candidate_score(pid, dist, mode, tp)
        results.append({
            "id": pid, "name": info["name"], "agent": info["agent"],
            "lat": info["lat"], "lon": info["lon"],
            "distance_m": int(dist), "price": price,
            "free_slots": sim.free(pid), "capacity": info["cap"],
            "occ_ratio": round(sim.occ_ratio(pid), 3),
            "pred_ratio": round(sim.pred_ratio(pid), 3),
            "score": round(sc, 4), "rank": 0,
        })
    results.sort(key=lambda x: x["score"])
    top5 = results[:5]
    for i, r in enumerate(top5): r["rank"] = i+1

    best_agent = min(AGENTS.items(), key=lambda a: haversine(b.lat,b.lon,a[1]["lat"],a[1]["lon"]))[0]

    with sim.lock:
        sim.total_assigned += 1
        decision = {
            "step": sim.step, "mode": mode,
            "parking": top5[0]["id"] if top5 else None,
            "agent":   top5[0]["agent"] if top5 else best_agent,
            "reward":  round(10*(1-top5[0]["occ_ratio"]), 2) if top5 else 0,
            "distance": top5[0]["distance_m"] if top5 else 0,
        }
        sim.recent_decisions.append(decision)
        if len(sim.recent_decisions) > 50: sim.recent_decisions.pop(0)

    return JSONResponse({
        "mode": mode, "agent": best_agent,
        "recommendation": top5[0] if top5 else None,
        "top5": top5, "step": sim.step,
        "traffic_pressure": round(tp, 3),
        "active_radius": max_dist,
        "pressure_regime": pressure_key,
    })

@app.get("/api/fl")
def fl_data(): return JSONResponse(FL_HISTORY)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1 — Recherche Intelligente (Inférence IA)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/search")
def search(b: SearchBody):
    mode = b.mode if b.mode in ["close","cheap","balanced"] else "balanced"
    tp   = sim.traffic_pressure
    TARGET_K = 5   # always return exactly 5 recommendations

    # 1. Premier passage : parkings dans le rayon ET sous le prix maximum
    candidates = []
    for pid, info in PARKINGS.items():
        if sim.free(pid) <= 0: continue
        if sim.pred_ratio(pid) >= 0.99: continue
        dist  = haversine(b.lat, b.lon, info["lat"], info["lon"])
        price = dyn_price(pid, mode, tp)
        in_radius = dist <= b.rayon_recherche
        in_budget = price <= b.prix_max
        candidates.append((pid, info, dist, price, in_radius, in_budget))

    # 2. Fallback progressif pour garantir exactement TARGET_K résultats
    #    Priorité 1 : dans le rayon + dans le budget
    #    Priorité 2 : hors rayon mais dans le budget  (on élargit le rayon)
    #    Priorité 3 : dans le rayon mais hors budget  (on ignore le prix)
    #    Priorité 4 : tous les parkings disponibles   (dernier recours)
    def build_pool(pool_filter):
        return [(pid, info, dist, price)
                for pid, info, dist, price, ir, ib in candidates
                if pool_filter(ir, ib)]

    pool = build_pool(lambda ir, ib: ir and ib)
    if len(pool) < TARGET_K:
        pool = build_pool(lambda ir, ib: ib)          # ignore radius
    if len(pool) < TARGET_K:
        pool = build_pool(lambda ir, ib: ir)          # ignore price
    if len(pool) < TARGET_K:
        pool = build_pool(lambda ir, ib: True)        # ignore both

    if not pool:
        return JSONResponse({
            "winner": None, "clusters": [], "top_k": [],
            "message": "No parking available in the system.",
            "search_params": {"rayon_m": b.rayon_recherche, "prix_max": b.prix_max, "mode": mode},
        })

    candidates = pool

    # 2. K-Means clustering sur les positions filtrées
    n_agents_present = len(set(info["agent"] for _, info, _, _ in candidates))
    k = max(1, min(n_agents_present, max(1, len(candidates)//3), 4))
    points = [(info["lat"], info["lon"]) for _, info, _, _ in candidates]
    labels = _kmeans(points, k)

    clusters: dict = {}
    for i, (pid, info, dist, price) in enumerate(candidates):
        c = labels[i]
        if c not in clusters:
            clusters[c] = {"cluster_id": c, "parkings": [], "agents": set(),
                           "centroid_lat": 0.0, "centroid_lon": 0.0}
        clusters[c]["parkings"].append(pid)
        clusters[c]["agents"].add(info["agent"])
        clusters[c]["centroid_lat"] += info["lat"]
        clusters[c]["centroid_lon"] += info["lon"]
    cluster_list = []
    for c, v in clusters.items():
        n = len(v["parkings"])
        cluster_list.append({
            "cluster_id": c,
            "parkings": v["parkings"],
            "agents": list(v["agents"]),
            "centroid_lat": round(v["centroid_lat"]/n, 5),
            "centroid_lon": round(v["centroid_lon"]/n, 5),
            "size": n,
        })

    # 3. Scoring DQN sur tous les candidats disponibles → toujours top 5
    scored = []
    for pid, info, dist, price in candidates:
        sc, p = candidate_score(pid, dist, mode, tp)
        scored.append({
            "id": pid, "name": info["name"], "agent": info["agent"],
            "lat": info["lat"], "lon": info["lon"],
            "distance_m": int(dist), "price": p,
            "free_slots": sim.free(pid), "capacity": info["cap"],
            "occ_ratio": round(sim.occ_ratio(pid), 3),
            "pred_ratio": round(sim.pred_ratio(pid), 3),
            "score": round(sc, 4),
        })
    scored.sort(key=lambda x: x["score"])
    top_k = scored[:TARGET_K]   # exactly 5 (or all available if < 5 parkings total)

    # 4. Décision DQN: re-scoring avec la fonction de récompense utilisateur
    prix_ref = max(b.prix_max, max((c["price"] for c in top_k), default=1.0))
    dist_ref = max(b.rayon_recherche, max((c["distance_m"] for c in top_k), default=1.0))
    def dqn_reward(c):
        price_util = 1.0 - (c["price"] / prix_ref)
        dist_util  = 1.0 - min(c["distance_m"] / dist_ref, 1.0)
        return 0.5 * price_util + 0.5 * dist_util - 0.3 * c["score"]

    winner = max(top_k, key=dqn_reward) if top_k else None
    if winner:
        winner = {
            **winner,
            "travel_time_min": round(winner["distance_m"] / 80, 1),
            "dqn_reward": round(dqn_reward(winner), 4),
        }

    return JSONResponse({
        "winner": winner,
        "clusters": cluster_list,
        "top_k": top_k,
        "search_params": {"rayon_m": b.rayon_recherche, "prix_max": b.prix_max, "mode": mode},
        "step": sim.step,
        "traffic_pressure": round(tp, 3),
    })

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2 — Télémétrie et État (Synchronisation temps réel)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/v1/parkings/status")
def parkings_status(lat: float, lon: float, radius_m: float = 1000.0):
    reserved_ids = {v["parking_id"] for v in RESERVATIONS.values() if v["status"] == "RÉSERVÉ"}
    with sim.lock:
        result = []
        for pid, info in PARKINGS.items():
            dist  = haversine(lat, lon, info["lat"], info["lon"])
            occ_r = sim.occupancy[pid] / info["cap"]
            reserved = pid in reserved_ids

            if reserved or occ_r >= 0.98:
                display_status = "checked"   # saturé ou réservé
            elif dist > radius_m:
                display_status = "grey"      # hors rayon
            elif occ_r < 0.40:
                display_status = "green"     # disponible (occupation optimale)
            else:
                display_status = "orange"    # occupation partielle

            result.append({
                "id": pid, "name": info["name"], "agent": info["agent"],
                "lat": info["lat"], "lon": info["lon"],
                "distance_m": int(dist),
                "in_radius": dist <= radius_m,
                "occ_ratio": round(occ_r, 3),
                "pred_ratio": round(sim.pred_ratio(pid), 3),
                "free": sim.free(pid),
                "capacity": info["cap"],
                "price": dyn_price(pid, "balanced", sim.traffic_pressure),
                "display_status": display_status,
                "reserved": reserved,
            })
        result.sort(key=lambda x: x["distance_m"])
        return JSONResponse({
            "parkings": result,
            "search_center": {"lat": lat, "lon": lon},
            "radius_m": radius_m,
            "step": sim.step,
        })

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3 — Apprentissage Fédéré (échange de poids)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/v1/federated/global_weights")
def global_weights():
    return JSONResponse({
        "status": "ok",
        "model": GLOBAL_MODEL,
        "message": (
            "Poids agrégés disponibles (FedAvg, 2 rounds, 4 clients). "
            "Initialisez votre DQN local avec ces métadonnées avant le prochain round."
        ),
    })

@app.post("/api/v1/federated/local_update")
def local_update(b: LocalUpdateBody):
    if b.agent_id not in AGENTS:
        raise HTTPException(status_code=404, detail=f"Agent inconnu: {b.agent_id}")
    if len(b.layer_norms) != len(GLOBAL_MODEL["layers"]):
        raise HTTPException(
            status_code=422,
            detail=f"layer_norms: attendu {len(GLOBAL_MODEL['layers'])} valeurs, reçu {len(b.layer_norms)}"
        )
    return JSONResponse({
        "agent_id": b.agent_id,
        "round": b.round,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "layer_norms_received": b.layer_norms,
        "status": "received",
        "message": (
            f"Poids de {b.agent_id} reçus (round {b.round}). "
            f"En attente des 3 autres agents pour agrégation FedAvg."
        ),
    }, status_code=202)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4 — Réservation et Rétroaction (Feedback Loop)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/v1/reservations")
def create_reservation(b: ReservationBody):
    if b.parking_id not in PARKINGS:
        raise HTTPException(status_code=404, detail=f"Parking inconnu: {b.parking_id}")
    if sim.free(b.parking_id) <= 0:
        raise HTTPException(status_code=409, detail="Parking complet — réservation impossible")

    rid   = str(uuid.uuid4())[:8].upper()
    info  = PARKINGS[b.parking_id]
    price = dyn_price(b.parking_id, b.mode, sim.traffic_pressure)
    dist  = haversine(b.lat, b.lon, info["lat"], info["lon"])

    with sim.lock:
        # Verrouille une place dans la simulation
        sim.occupancy[b.parking_id] = min(info["cap"], sim.occupancy[b.parking_id] + 1)

    RESERVATIONS[rid] = {
        "reservation_id": rid,
        "parking_id": b.parking_id,
        "parking_name": info["name"],
        "status": "RÉSERVÉ",
        "lat_user": b.lat,
        "lon_user": b.lon,
        "distance_m": int(dist),
        "price_eur_h": price,
        "mode": b.mode,
        "agent": info["agent"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "feedback_submitted": False,
    }
    return JSONResponse(RESERVATIONS[rid], status_code=201)

@app.post("/api/v1/feedback")
def submit_feedback(b: FeedbackBody):
    if b.reservation_id not in RESERVATIONS:
        raise HTTPException(status_code=404, detail=f"Réservation inconnue: {b.reservation_id}")
    if b.action not in ("ACCEPTER", "REFUSER"):
        raise HTTPException(status_code=422, detail="action doit être 'ACCEPTER' ou 'REFUSER'")
    if not (1 <= b.rating <= 5):
        raise HTTPException(status_code=422, detail="rating doit être entre 1 et 5")

    res = RESERVATIONS[b.reservation_id]
    if res["feedback_submitted"]:
        raise HTTPException(status_code=409, detail="Feedback déjà soumis pour cette réservation")

    # Signal de récompense RL: négatif si refus ou mauvaise note
    if b.action == "REFUSER":
        reward_delta = -2.0
    elif b.rating <= 2:
        reward_delta = -1.5 + (b.rating - 1) * 0.5   # −1.5 ou −1.0
    elif b.rating == 3:
        reward_delta = 0.0
    else:
        reward_delta = 0.5 * (b.rating - 3)           # +0.5 ou +1.0

    rl_signal = "PENALITE" if reward_delta < 0 else "BONUS" if reward_delta > 0 else "NEUTRE"
    entry = {
        "reservation_id": b.reservation_id,
        "parking_id": res["parking_id"],
        "agent_id": res["agent"],
        "action": b.action,
        "rating": b.rating,
        "reward_delta": round(reward_delta, 2),
        "mode": res["mode"],
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "rl_signal": rl_signal,
    }
    RL_FEEDBACK_LOG.append(entry)

    RESERVATIONS[b.reservation_id]["feedback_submitted"] = True
    RESERVATIONS[b.reservation_id]["status"] = "COMPLÉTÉ" if b.action == "ACCEPTER" else "REFUSÉ"

    msg = (
        f"Signal RL ({rl_signal}: {reward_delta:+.2f}) injecté dans la mémoire de {res['agent']}. "
        "Exploration forcée lors du prochain épisode."
        if reward_delta < 0 else
        f"Feedback positif enregistré. Récompense: {reward_delta:+.2f} pour {res['agent']}."
    )
    return JSONResponse({**entry, "message": msg})

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn, os
    port = int(os.environ.get("PORT", 8000))
    print("\n" + "="*55)
    print("  SmartParking AI — Edge V2X Prototype")
    print(f"  Ouvrez votre navigateur sur : http://localhost:{port}")
    print("="*55 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
