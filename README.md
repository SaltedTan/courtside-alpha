# Courtside Alpha

Quantitative prediction market engine that compares a real-time NBA win-probability model against Polymarket odds, with testnet order signing for shadow trading.

**Stack:** Python (FastAPI, XGBoost, scikit-learn) &middot; Rust (Tokio, Axum) &middot; TypeScript (Next.js, Recharts) &middot; SQLite

## Quick start

```bash
./start.sh              # launches all 4 services
open http://localhost:3000  # open the dashboard
./start.sh --stop       # shut everything down
```

Logs are written to `./logs/`. For manual per-service startup, see below.

---

## Architecture

```
Polymarket WS ──► Execution Engine (Rust:4000) ──► SQLite (trades.sqlite)
                        │         │                        ▲
                        │         └──► Alpha Engine         │
                        │              POST /predict        │
                        │              (Python:8001)        │
                        │                                   │
                        └──► Live Game Server          Dashboard (Next.js:3000)
                              GET /games                GET /trades, /wallet
                              (Python:8000)
```

Four services form a real-time pipeline:

1. **Live Game Server** polls the NBA live API for scores and boxscore stats every 30 s
2. **Alpha Engine** loads a 4-model XGBoost ensemble (win probability, margin, market proxy, edge confidence) and serves 188-feature predictions over HTTP
3. **Execution Engine** (Rust) connects to the Polymarket WebSocket, queries the Alpha Engine on each price tick, sizes positions via Kelly criterion, and signs EIP-712 orders with a testnet key
4. **Dashboard** polls the Execution Engine every 3 s and renders live trades, wallet state, and model-vs-market charts

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | >= 3.11 |
| Rust + Cargo | >= 1.78 |
| Node.js | >= 20 |

---

## One-time setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Tip:** Use `./venv/bin/python` directly to avoid Anaconda or system Python shadowing the venv.

---

## Running services individually

### 1 — Live Game Server (Python, port 8000)

```bash
./venv/bin/python -m uvicorn live_server.app:app --port 8000
```

### 2 — Alpha Engine (Python, port 8001)

```bash
./venv/bin/python alpha_engine/app.py
```

### 3 — Execution Engine (Rust, port 4000)

```bash
cd execution-engine && cargo run --release
```

| Env var | Default | Description |
|---------|---------|-------------|
| `RUST_LOG` | `info` | Log level (`debug` for verbose output) |

### 4 — Dashboard (Next.js, port 3000)

```bash
cd dashboard && npm install && npm run dev
```

---

## Project structure

```
courtside-alpha/
├── features.py             # Shared 188-feature engine (team profiles, live stats, lag features)
├── requirements.txt        # Python dependencies
├── start.sh                # One-command launcher for all 4 services
│
├── live_server/            # Live game state server (port 8000)
│   ├── app.py              #   FastAPI app — polls NBA API, runs models, exposes signals
│   ├── market_data.py      #   Polymarket + Kalshi odds client
│   └── recorder.py         #   SQLite observation recorder for retraining
│
├── alpha_engine/           # ML inference server (port 8001)
│   └── app.py              #   FastAPI app — stateless prediction endpoint
│
├── execution-engine/       # Rust shadow trader (port 4000)
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs         #   WebSocket client, Kelly sizing, trade logging
│       └── wallet.rs       #   EIP-712 order signing (secp256k1)
│
├── dashboard/              # Next.js real-time UI (port 3000)
│   └── src/app/
│       ├── page.tsx        #   Main dashboard component
│       ├── types.ts        #   Shared TypeScript interfaces
│       ├── utils.ts        #   Helper functions
│       └── components/     #   UI components (charts, cards, feed)
│
├── training/               # Offline: data collection & model training
│   ├── train_model.py      #   XGBoost training pipeline (4-model ensemble)
│   ├── fetch_games.py      #   NBA API historical data collector
│   ├── fetch_pbp.py        #   Play-by-play scraper (S3)
│   └── fetch_boxscores.py  #   Advanced boxscore scraper (S3)
│
├── data/                   # Model artifacts & team profiles (parquet + JSON)
└── .env.example
```

---

## Key API endpoints

| Service | Endpoint | Description |
|---------|----------|-------------|
| Live Server | `GET /games` | All live games with predictions |
| Live Server | `GET /signals` | Games with active trading signals |
| Live Server | `GET /health` | Service health check |
| Alpha Engine | `POST /predict` | Stateless prediction from game state |
| Execution Engine | `GET /trades` | All shadow trades |
| Execution Engine | `GET /wallet` | Testnet wallet state |
