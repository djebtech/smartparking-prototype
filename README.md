# Smart Parking Prototype — V2X + Edge AI

Prototype interactif du système de gestion dynamique des places de stationnement urbain basé sur le Dueling DQN et le Federated Learning (Flower).

## Lancement en 3 étapes

### 1. Installer les dépendances

```bash
cd "donya memoire/new/prototype"
pip install -r requirements.txt
```

### 2. Démarrer le serveur

```bash
python app.py
```

### 3. Ouvrir dans le navigateur

```
http://localhost:8000
```

Accessible depuis n'importe quel appareil sur le même réseau local via :
```
http://<votre-IP>:8000
```

---

## Fonctionnalités

| Tab | Description |
|-----|-------------|
| 🗺️ **Carte Live** | Carte Leaflet avec zones agents (cercles), 28 parkings colorés selon occupation, sidebar avec barres de progression |
| 🅿️ **Recommandation** | Cliquez sur la carte pour positionner un véhicule, choisissez un mode (proche / moins cher / équilibré), obtenez les 5 meilleurs parkings via le moteur DQN |
| 📊 **Métriques** | 5 graphiques en temps réel : occupation, récompenses, assignations, convergence de la loss, occupation par parking |
| 🤖 **Federated Learning** | Architecture FL, métriques par round, réduction de loss par agent, courbe epsilon, tableau complet des résultats |

---

## Architecture technique

```
Frontend (Leaflet.js + Chart.js)
        ↕ REST API (polling 2s)
Backend FastAPI (app.py)
    ├── /api/config      → configuration statique (agents, parkings)
    ├── /api/state       → état en temps réel (occupations simulées)
    ├── /api/recommend   → moteur de recommandation DQN (Top-K scoring)
    └── /api/fl          → historique Federated Learning (données réelles)
```

---

## Données réelles utilisées

- **Scénario SUMO** : Luxembourg LuST (28 parkings, capacité totale ~2000 places)
- **Résultats FL** : 2 rounds × 4 agents
  - Round 1 : reward moyen = 4497.63, loss = 0.2189
  - Round 2 : reward moyen = 4572.64, loss = 0.1665 (−23.9%)
- **Taux d'assignation global** : 99.4% (577.5 véhicules/step)

---

## Structure du projet

```
prototype/
├── app.py              ← Backend FastAPI
├── requirements.txt    ← Dépendances Python
├── README.md           ← Ce fichier
└── static/
    └── index.html      ← Frontend complet (Leaflet + Chart.js)
```
