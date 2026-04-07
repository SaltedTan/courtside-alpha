"""
Fetch Historical Play-by-Play Data
====================================
Pulls PBP for games missing from data/play_by_play.parquet using the NBA
S3 live data endpoint (same source as fetch_boxscores.py — no auth, no blocking).

Output rows are identical in schema to what nba_data.py already produced,
so they can be appended directly to the existing parquet.

Usage:
    python fetch_pbp.py

Takes ~5-10 minutes for 359 games. Safe to re-run — skips already-fetched games.
"""

import pandas as pd
import numpy as np
import requests
import time
import os

DATA_DIR = "data"
PBP_PATH = f"{DATA_DIR}/play_by_play.parquet"

PBP_URL = (
    "https://nba-prod-us-east-1-mediaops-stats.s3.amazonaws.com"
    "/NBA/liveData/playbyplay/playbyplay_{game_id}.json"
)

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
})


def parse_clock(clock_str):
    """Convert 'MM:SS' to integer seconds."""
    try:
        parts = str(clock_str).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0


def clock_to_pctimestring(clock_iso):
    """Convert 'PT12M00.00S' → '12:00'."""
    if not clock_iso:
        return "0:00"
    s = str(clock_iso)
    if s.startswith("PT"):
        s = s[2:]
        minutes = 0
        secs = 0
        if "M" in s:
            m_part, s = s.split("M", 1)
            minutes = int(m_part)
        if "S" in s:
            s_part = s.replace("S", "")
            if s_part:
                secs = int(float(s_part))
        return f"{minutes}:{secs:02d}"
    return "0:00"


def fetch_game_pbp(game_id):
    """Fetch PBP for a single game. Returns a DataFrame or None."""
    url = PBP_URL.format(game_id=game_id)
    try:
        resp = SESSION.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Fetch failed for {game_id}: {e}")
        return None

    actions = data.get("game", {}).get("actions", [])
    if not actions:
        return None

    rows = []
    for a in actions:
        period = int(a.get("period", 1))
        clock_iso = a.get("clock", "PT0M0.00S")
        pct = clock_to_pctimestring(clock_iso)
        secs_rem = parse_clock(pct)
        game_secs_left = max(0, (4 - period)) * 720 + secs_rem

        score_home = pd.to_numeric(a.get("scoreHome", None), errors="coerce")
        score_away = pd.to_numeric(a.get("scoreAway", None), errors="coerce")
        home_margin = (score_home - score_away) if pd.notna(score_home) and pd.notna(score_away) else np.nan

        rows.append({
            "gameId": a.get("gameId", game_id),
            "actionNumber": a.get("actionNumber"),
            "clock": clock_iso,
            "period": period,
            "teamId": a.get("teamId"),
            "teamTricode": a.get("teamTricode", ""),
            "personId": a.get("personId"),
            "playerName": a.get("playerName", ""),
            "playerNameI": a.get("playerNameI", ""),
            "xLegacy": a.get("xLegacy"),
            "yLegacy": a.get("yLegacy"),
            "shotDistance": a.get("shotDistance"),
            "shotResult": a.get("shotResult", ""),
            "isFieldGoal": a.get("isFieldGoal"),
            "scoreHome": a.get("scoreHome", ""),
            "scoreAway": a.get("scoreAway", ""),
            "pointsTotal": a.get("pointsTotal"),
            "location": a.get("location", ""),
            "description": a.get("description", ""),
            "actionType": a.get("actionType", ""),
            "subType": a.get("subType", ""),
            "videoAvailable": a.get("videoAvailable"),
            "shotValue": a.get("shotValue"),
            "actionId": a.get("actionId"),
            # Derived columns matching nba_data.py output
            "PCTIMESTRING": pct,
            "SECONDS_REMAINING": secs_rem,
            "PERIOD": period,
            "GAME_SECONDS_LEFT": game_secs_left,
            "SCOREHOME": score_home,
            "SCOREAWAY": score_away,
            "HOME_MARGIN": home_margin,
            "GAME_ID": game_id,
        })

    return pd.DataFrame(rows)


def main():
    # Load existing PBP
    if not os.path.exists(PBP_PATH):
        print("No existing play_by_play.parquet found — run nba_data.py first.")
        return

    existing = pd.read_parquet(PBP_PATH)
    done_ids = set(existing["GAME_ID"].unique())
    print(f"Existing PBP: {len(done_ids)} games, {len(existing)} rows")

    # Find missing games from season_games
    games = pd.read_parquet(f"{DATA_DIR}/season_games.parquet")
    all_ids = sorted(games["GAME_ID"].unique())
    missing = [gid for gid in all_ids if gid not in done_ids]
    print(f"Missing: {len(missing)} games to fetch\n")

    if not missing:
        print("All PBP already fetched!")
        return

    # Quick connectivity test
    print(f"Testing with {missing[0]}...")
    test_df = fetch_game_pbp(missing[0])
    if test_df is None or test_df.empty:
        print("  FAILED — S3 endpoint unreachable or game not found")
        # Try next game
        for gid in missing[1:4]:
            test_df = fetch_game_pbp(gid)
            if test_df is not None and not test_df.empty:
                print(f"  OK with {gid}: {len(test_df)} plays")
                missing = [g for g in missing if g != gid]
                missing.insert(0, gid)
                break
        else:
            print("  All tests failed — check connectivity")
            return
    else:
        print(f"  OK: {missing[0]} — {len(test_df)} plays")

    new_dfs = [test_df]
    failed = []
    remaining = missing[1:]

    for i, game_id in enumerate(remaining):
        df = fetch_game_pbp(game_id)

        if df is not None and not df.empty:
            new_dfs.append(df)
        else:
            failed.append(game_id)

        # Checkpoint every 50 games
        if (i + 1) % 50 == 0:
            new_combined = pd.concat(new_dfs, ignore_index=True)
            full = pd.concat([existing, new_combined], ignore_index=True)
            full.to_parquet(PBP_PATH, index=False)
            total_games = full["GAME_ID"].nunique()
            print(f"  [{i+2}/{len(remaining)+1}] checkpoint: {total_games} games total, {len(failed)} failed")

        time.sleep(0.3)

    # Final save
    if new_dfs:
        new_combined = pd.concat(new_dfs, ignore_index=True)
        full = pd.concat([existing, new_combined], ignore_index=True)
        full.to_parquet(PBP_PATH, index=False)
        print(f"\nDone! {full['GAME_ID'].nunique()} total games, {len(full)} rows")
        print(f"New games added: {len(new_dfs)}")
    if failed:
        print(f"Failed ({len(failed)}): {failed[:20]}{'...' if len(failed) > 20 else ''}")


if __name__ == "__main__":
    main()
