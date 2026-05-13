# CLAUDE.md — SmartParking AI · Edge V2X Prototype

> Full context for Claude Code. Read this before touching any file.

---

## 1. What this project is

This is the **interactive web prototype** for a Master's thesis at USTHB (Algeria):

> **"Optimisation de la gestion dynamique des places de stationnement urbain à l'aide de l'Intelligence Artificielle embarquée en Edge dans un environnement V2X"**

The real system runs on:
- **SUMO** (Luxembourg LuST scenario, 28 parkings, ~2000 capacity)
- **Dueling DQN** per agent (state=48 dims, action=5 candidates)
- **Federated Learning** via Flower (FedAvg, 4 clients, 2 rounds, gRPC:8080)
- **V2X communication** layer (V2I messages, zone state broadcasts)
- **Edge computing** — each agent is autonomous, no cloud

This prototype **replaces SUMO/TraCI** with a FastAPI backend that simulates the same logic and exposes it as a REST API consumed by a single-page HTML/JS frontend.

---

## 2. Project structure

```
prototype/
├── CLAUDE.md              ← you are here
├── app.py                 ← FastAPI backend (all logic)
├── requirements.txt       ← fastapi, uvicorn[standard]
├── README.md              ← user-facing run instructions
└── static/
    └── index.html         ← full SPA frontend (~700 lines)
```

The real thesis source code lives one level up at:
```
../../../                  ← /Users/macbook/Desktop/donya/
  multi_agent_env.py       ← RL environment (state/reward/step)
  dqn.py                   ← Dueling DQN architecture
  parking_manager.py       ← dynamic pricing, TraCI assignment
  client_flower.py         ← Flower FL client (fit/evaluate)
  server_flower.py         ← Flower FL server (FedAvg)
  v2x_comm.py              ← V2X message types + feature extractor
  replay.py                ← Priority Replay Buffer
  agent_mapper.py          ← weighted K-Means zone assignment
  build_weighted_agent_clusters.py
  agents_weighted_kmeans_balanced.json  ← 4 agent configs
  parkings_min300.add.xml  ← 28 parkings definition
  server_flower.log        ← REAL training results (used in app.py)
  agent_1.log … agent_4.log
```

---

## 3. How to run

```bash
cd "donya memoire/new/prototype"
pip3 install -r requirements.txt
python3 app.py
# → open http://localhost:8000
```

---

## 4. Backend — app.py

### Architecture

```
FastAPI app
  ├── AGENTS dict       (4 agents, GPS coords, zone config)
  ├── PARKINGS dict     (28 parkings P1–P28, GPS, capacity, agent)
  ├── FL_HISTORY dict   (real Flower training results)
  ├── SimState class    (live occupancy simulation, background thread)
  ├── haversine()       (GPS distance in meters)
  ├── dyn_price()       (dynamic pricing — replica of parking_manager.py)
  ├── candidate_score() (DQN scoring — replica of multi_agent_env.py)
  └── Routes:
        GET  /              → serves static/index.html
        GET  /api/config    → static agents + parkings config
        GET  /api/state     → live occupancy snapshot (polled every 2s)
        POST /api/recommend → DQN recommendation engine
        GET  /api/fl        → FL training history
```

### SimState (live simulation)

- Background thread ticks every 2 seconds
- Each tick: updates occupancy (±random), recalculates incoming vehicles
- Traffic pressure cycles: peaks at step 200–450 (1.30–1.40) and 700–950 (1.38–1.50), baseline 1.00–1.15
- `sim.lock` (threading.Lock) protects all state reads/writes

### Dynamic pricing formula (exact replica from parking_manager.py)

```python
p = 2.0 + 0.80 * occ_ratio + 0.60 * pred_ratio
p += mode_adjustment  # cheap: -0.30, close: +0.25, balanced: +0.10
p *= min(max(traffic_pressure, 0.9), 1.3)
# saturation surcharges:
if pred_ratio >= 0.90: p += 1.50
elif pred_ratio >= 0.75: p += 0.90
elif pred_ratio >= 0.60: p += 0.45
p = clamp(p, 1.5, 7.0)
```

### Candidate scoring formula (exact replica from multi_agent_env.py)

```python
# Normalized inputs
dist_norm  = min(dist / 3000, 1.0)
price_norm = min(price / 6.5,  1.0)
incom_norm = min(incoming / cap, 1.0)
free_ratio = free / cap

# Mode weights
if mode == "close":    s = 1.65*dn + 0.10*pn + 0.18*pred_r + 0.05*ir - 0.08*fr
elif mode == "cheap":  s = 0.45*dn + 1.30*pn + 0.18*pred_r + 0.05*ir - 0.08*fr
else:                  s = 0.75*dn + 0.75*pn + 0.22*pred_r + 0.05*ir - 0.10*fr

# Saturation penalty (lower score = better)
if pred_r > 0.60: s += 0.25
if pred_r > 0.80: s += 0.70
if pred_r > 0.92: s += 1.60
```

Results are sorted ascending (lowest score = best candidate), top 5 returned.

### /api/recommend response shape

```json
{
  "mode": "balanced",
  "agent": "agent_2",
  "recommendation": {
    "id": "P5", "name": "P5 – Bonnevoie", "agent": "agent_2",
    "lat": 49.617, "lon": 6.135,
    "distance_m": 264, "price": 2.58,
    "free_slots": 74, "capacity": 100,
    "occ_ratio": 0.22, "pred_ratio": 0.26,
    "score": 0.3491, "rank": 1
  },
  "top5": [ ...same shape, rank 1–5... ],
  "step": 42,
  "traffic_pressure": 1.12
}
```

### /api/fl response shape

```json
{
  "rounds": [
    { "round": 1, "epsilon": 0.900,
      "fit":  { "reward": 4497.63, "assigned": 568.5, "loss": 0.2189, "buffer": 573.5, "steps": 573.5 },
      "eval": { "reward": 1471.90, "assigned": 145.25 } },
    { "round": 2, "epsilon": 0.720,
      "fit":  { "reward": 4572.64, "assigned": 577.5, "loss": 0.1665, "buffer": 1154.5, "steps": 581.0 },
      "eval": { "reward": 1464.30, "assigned": 143.50 } }
  ],
  "per_agent": {
    "agent_1": {
      "fit":  [ {"round":1,"reward":2861.03,"assigned":289,"loss":0.2806,"steps":294},
                {"round":2,"reward":2866.38,"assigned":300,"loss":0.2003,"steps":308} ],
      "eval": [ {"round":1,"reward":791.94,"assigned":81},
                {"round":2,"reward":703.85,"assigned":71} ]
    },
    ... (agent_2, agent_3, agent_4 same shape)
  }
}
```

### Agent config (from agents_weighted_kmeans_balanced.json)

| Agent | Name | Centroid (SUMO x,y) | GPS center | Parkings | Capacity |
|---|---|---|---|---|---|
| agent_1 | Zone Nord-Est | (8776, 8428) | 49.635, 6.170 | P1,P2,P3,P19–P22 | 540 |
| agent_2 | Zone Centre | (5848, 6352) | 49.619, 6.133 | P4,P5,P12,P15,P23,P24,P28 | 500 |
| agent_3 | Zone Sud | (6604, 4779) | 49.607, 6.142 | P6,P7,P8,P10,P11,P27 | 480 |
| agent_4 | Zone Ouest | (3684, 7089) | 49.625, 6.106 | P9,P13–P18,P25,P26 | 480 |

GPS coordinate mapping from SUMO Cartesian:
```
lat = 49.570 + (y / 11000) * 0.085
lon = 6.060  + (x / 12000) * 0.150
```

---

## 5. Frontend — static/index.html

Single-file SPA, no build step, no npm. Pure HTML + Leaflet.js + Chart.js.

### Tabs

| Tab | ID | Description |
|---|---|---|
| Carte Live | `tab-map` | Leaflet map, agent zone circles, 28 parking markers, sidebar list |
| Recommandation | `tab-rec` | Click-to-place vehicle map, mode selector, Top-5 result cards |
| Métriques | `tab-metrics` | 5 Chart.js charts (occupancy, rewards, assignments, loss, per-parking) |
| Federated Learning | `tab-fl` | FL architecture diagram, round selector, loss bars, epsilon chart, results table |

### Key global variables

```javascript
let cfg = null;        // /api/config response (agents + parkings)
let state = null;      // /api/state response (live)
let flData = null;     // /api/fl response
let mapLive = null;    // Leaflet map instance (tab 1)
let mapRec  = null;    // Leaflet map instance (tab 2)
let recPos  = null;    // {lat, lon} of placed vehicle
let charts  = {};      // Chart.js instances keyed by name
```

### Polling

`startPolling()` calls `/api/state` every 2000ms → triggers `updateMapMarkers()`, `updateSidebar()`, `updateStatBar()`, `updateCharts()`.

### Parking marker colors

```
status "low"    → green  (#22c55e)
status "medium" → orange (#f59e0b)
status "high"   → red    (#ef4444)
status "full"   → dark   (#374151)
```

### Chart.js instances

| Key | Type | Data source |
|---|---|---|
| `occ` | line | rolling occupancy % from /api/state |
| `rewards` | bar | per-agent fit rewards from flData |
| `assignments` | line | per-agent fit assignments from flData |
| `loss` | line | per-agent loss R1→R2 from flData |
| `parkingOcc` | bar | per-parking occ_ratio from /api/state |

---

## 6. Real training results (from server_flower.log + agent logs)

| Metric | Round 1 | Round 2 | Δ |
|---|---|---|---|
| Avg reward | 4 497.63 | 4 572.64 | +1.7% |
| Avg assigned | 568.5 | 577.5 | +1.6% |
| Avg loss | 0.2189 | 0.1665 | −23.9% |
| Runtime | — | — | 12 936.70s total |

| Agent | R1 reward | R2 reward | R1 loss | R2 loss |
|---|---|---|---|---|
| agent_1 | 2 861.03 | 2 866.38 | 0.2806 | 0.2003 |
| agent_2 | 5 276.28 | 5 151.99 | 0.1735 | 0.1496 |
| agent_3 | 6 374.95 | 6 585.19 | 0.1804 | 0.1465 |
| agent_4 | 3 478.25 | 3 686.99 | 0.2413 | 0.1695 |

Global assignment rate: **99.4%** (577.5 vehicles/step average).

---

## 7. Key design decisions

- **No SUMO in prototype**: TraCI requires the full SUMO binary running. The prototype replicates only the pure Python scoring logic (pricing + candidate scoring) without any simulation coupling. Background thread simulates realistic occupancy drift.
- **Single HTML file**: No React, no build pipeline. The frontend is `static/index.html` — everything inline. Keeps deployment to `python3 app.py`.
- **GPS approximation**: SUMO uses local Cartesian coords (0–12000m × 0–11000m). The GPS mapping is a linear approximation centered on Luxembourg city. Visually correct for demo purposes.
- **Score sort order**: Lower score = better. The candidate_score function penalizes distant, expensive, and saturated parkings. Results are `.sort(key=lambda x: x["score"])` ascending.

---

## 8. Possible next features (not yet built)

- [ ] **User authentication** — save vehicle position history per user
- [ ] **WebSocket** instead of polling — push state updates from server
- [ ] **Map clustering** — group nearby parkings at low zoom levels
- [ ] **Heatmap layer** — occupancy density visualization on map
- [ ] **More FL rounds** — simulate additional rounds with extrapolated metrics
- [ ] **Mode comparison view** — show Top-5 for all 3 modes side by side
- [ ] **Mobile responsive** — optimize layout for phone screens
- [ ] **Export** — download recommendation history as CSV
- [ ] **Agent detail panel** — click an agent circle to see its zone stats
- [ ] **Replay simulation** — replay a full episode from the logs step by step
- [ ] **Dark/light mode toggle**

---

## 9. Dependencies

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
```

Frontend CDN libraries (no install needed):
- Leaflet.js 1.9.4 — `https://unpkg.com/leaflet@1.9.4/`
- Chart.js 4.4.0 — `https://cdn.jsdelivr.net/npm/chart.js@4.4.0/`
- OpenStreetMap tiles — `https://{s}.tile.openstreetmap.org/`

---

## 10. Author context

- **Student**: Djebrane (djebtech@gmail.com)
- **Program**: Master Télécommunications, USTHB Bab Ezzouar, Algeria
- **Thesis supervisor**: [not specified]
- **Defense**: 2026
- **Thesis folder**: `/Users/macbook/Desktop/donya/donya memoire/`
- **Source code**: `/Users/macbook/Desktop/donya/`
- **This prototype**: `/Users/macbook/Desktop/donya/donya memoire/new/prototype/`
