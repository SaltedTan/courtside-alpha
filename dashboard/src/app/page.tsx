"use client";

import { useEffect, useRef, useState } from "react";
import type { Trade, WalletInfo, LiveGame, Tab, FeedEvent, ToastNotification } from "./types";
import { alignedProbs, fmt, shortHash, betTeam, timeSince } from "./utils";
import { StatCard, DataPoint, ActivePositionCard, LiveGameCard, ActionWinRateCard } from "./components/cards";
import { SectionHeader, StatusBadge, LivePipelineFeed, ToastStack } from "./components/ui";
import { ProbabilityChart, EdgeChart, SparklineChart } from "./components/charts";

export default function Dashboard() {
  const [trades, setTrades]           = useState<Trade[]>([]);
  const [wallet, setWallet]           = useState<WalletInfo | null>(null);
  const [liveGames, setLiveGames]     = useState<LiveGame[]>([]);
  const [loading, setLoading]         = useState(true);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [connected, setConnected]     = useState(false);
  const [activeTab, setActiveTab]     = useState<Tab>("overview");
  const [feedEvents, setFeedEvents]   = useState<FeedEvent[]>([]);
  const [toasts, setToasts]           = useState<ToastNotification[]>([]);
  const prevTradesRef  = useRef<Trade[]>([]);
  const prevGamesRef   = useRef<LiveGame[]>([]);
  const isFirstPollRef = useRef(true);

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [tradesRes, walletRes] = await Promise.all([
          fetch("/api/trades"),
          fetch("/api/wallet"),
        ]);
        const newTrades: Trade[] = await tradesRes.json();
        setTrades(newTrades);
        setWallet(await walletRes.json());

        let newGames: LiveGame[] = [];
        try {
          const gamesRes = await fetch("/api/games");
          if (gamesRes.ok) {
            const data = await gamesRes.json();
            newGames = data.games ?? [];
            setLiveGames(newGames);
          }
        } catch { /* server may not be running */ }

        const now = new Date();

        if (!isFirstPollRef.current) {
          const newEvents: FeedEvent[] = [];

          const prevIds = new Set(prevTradesRef.current.map(t => t.id));
          for (const trade of newTrades) {
            if (!prevIds.has(trade.id)) {
              const edge = (trade.model_implied_prob - trade.market_implied_prob) * 100;
              newEvents.push({
                id:        `trade-${trade.id}`,
                type:      "TRADE",
                message:   `${trade.action}  ${trade.target_team}`,
                detail:    `${edge >= 0 ? "+" : ""}${edge.toFixed(1)}% edge \u00b7 $${trade.stake_amount} staked`,
                timestamp: now,
              });
              const toastId = `toast-${trade.id}`;
              setToasts(prev => [...prev, { id: toastId, trade }]);
              setTimeout(() => setToasts(prev => prev.filter(t => t.id !== toastId)), 6000);
            }
          }

          const prevGamesMap = new Map(prevGamesRef.current.map(g => [g.game_id, g]));
          for (const game of newGames) {
            const prev = prevGamesMap.get(game.game_id);
            if (!prev) {
              newEvents.push({
                id:        `nba-new-${game.game_id}-${now.getTime()}`,
                type:      "NBA_DATA",
                message:   `${game.home_team} vs ${game.away_team}`,
                detail:    `New game tracked \u00b7 Q${game.period} \u00b7 ${game.score.home}\u2013${game.score.away}`,
                timestamp: now,
              });
              newEvents.push({
                id:        `pred-new-${game.game_id}-${now.getTime()}`,
                type:      "ML_PREDICTION",
                message:   `${game.home_team} win prob ${(game.predictions.win_probability * 100).toFixed(1)}%`,
                detail:    `edge ${game.predictions.edge >= 0 ? "+" : ""}${(game.predictions.edge * 100).toFixed(1)}% \u00b7 conf ${(game.predictions.edge_confidence * 100).toFixed(0)}%`,
                timestamp: now,
              });
            } else {
              if (prev.score.home !== game.score.home || prev.score.away !== game.score.away) {
                newEvents.push({
                  id:        `nba-score-${game.game_id}-${now.getTime()}`,
                  type:      "NBA_DATA",
                  message:   `${game.home_team} ${game.score.home}\u2013${game.score.away} ${game.away_team}`,
                  detail:    `Q${game.period} ${game.game_clock} \u00b7 score update`,
                  timestamp: now,
                });
              }
              if (Math.abs(game.predictions.win_probability - prev.predictions.win_probability) > 0.005) {
                const edgePct = game.predictions.edge * 100;
                newEvents.push({
                  id:        `pred-${game.game_id}-${now.getTime()}`,
                  type:      "ML_PREDICTION",
                  message:   `${game.home_team} win prob \u2192 ${(game.predictions.win_probability * 100).toFixed(1)}%`,
                  detail:    `edge ${edgePct >= 0 ? "+" : ""}${edgePct.toFixed(1)}% \u00b7 conf ${(game.predictions.edge_confidence * 100).toFixed(0)}%`,
                  timestamp: now,
                });
              }
            }
          }

          if (newEvents.length > 0) {
            setFeedEvents(prev => [...newEvents, ...prev].slice(0, 50));
          }
        } else {
          isFirstPollRef.current = false;
          const initEvents: FeedEvent[] = [];
          if (newGames.length > 0) {
            initEvents.push({
              id:        `init-pred-${now.getTime()}`,
              type:      "ML_PREDICTION",
              message:   `XGBoost models active \u00b7 ${newGames.length} game${newGames.length > 1 ? "s" : ""} tracked`,
              detail:    `265 features \u00b7 4 models per game`,
              timestamp: now,
            });
            initEvents.push({
              id:        `init-games-${now.getTime()}`,
              type:      "NBA_DATA",
              message:   newGames.map(g => `${g.home_team} vs ${g.away_team}`).join(" \u00b7 "),
              detail:    `NBA live scoreboard \u00b7 polling every 30s`,
              timestamp: now,
            });
          }
          if (newTrades.length > 0) {
            initEvents.push({
              id:        `init-trades-${now.getTime()}`,
              type:      "TRADE",
              message:   `${newTrades.length} trade${newTrades.length > 1 ? "s" : ""} on record`,
              detail:    `Latest: ${newTrades[0]?.target_team ?? "\u2014"} \u00b7 ${newTrades[0]?.action ?? ""}`,
              timestamp: now,
            });
          }
          if (initEvents.length > 0) setFeedEvents(initEvents);
        }

        prevTradesRef.current = newTrades;
        prevGamesRef.current  = newGames;
        setLastUpdated(now);
        setConnected(true);
      } catch {
        setConnected(false);
      } finally {
        setLoading(false);
      }
    };

    fetchAll();
    const interval = setInterval(fetchAll, 3000);
    return () => clearInterval(interval);
  }, []);

  // ── Derived stats ──
  const isWon  = (t: Trade) => t.status === "WON"  || (t.status === "CLOSED" && t.pnl != null && t.pnl > 0);
  const isLost = (t: Trade) => t.status === "LOST" || (t.status === "CLOSED" && t.pnl != null && t.pnl < 0);
  const won         = trades.filter(isWon).length;
  const lost        = trades.filter(isLost).length;
  const open        = trades.filter(t => t.status === "OPEN").length;
  const resolved    = won + lost;
  const winRate     = resolved > 0 ? (won / resolved) * 100 : 0;
  const totalPnl    = trades.reduce((s, t) => s + (t.pnl ?? 0), 0);
  const totalStaked = trades.reduce((s, t) => s + t.stake_amount, 0);
  const avgEdge     = trades.length > 0
    ? trades.reduce((s, t) => s + Math.abs(alignedProbs(t).edge), 0) / trades.length * 100
    : null;
  const signedCount = trades.filter(t => t.order_hash).length;

  const roi      = totalStaked > 0 ? (totalPnl / totalStaked) * 100 : null;
  const avgStake = trades.length > 0 ? totalStaked / trades.length : null;

  const resolvedOrdered = trades.filter(t => t.status === "WON" || t.status === "LOST" || t.status === "CLOSED");
  let streakCount = 0;
  let streakType: "W" | "L" | null = null;
  for (const t of resolvedOrdered) {
    const s = isWon(t) ? "W" : "L";
    if (streakType === null) { streakType = s; streakCount = 1; }
    else if (s === streakType) { streakCount++; }
    else break;
  }

  const resolvedWithPnl = trades.filter(t => t.pnl != null);
  const bestTrade  = resolvedWithPnl.length > 0 ? resolvedWithPnl.reduce((b, t) => t.pnl! > b.pnl! ? t : b) : null;
  const lostTrades = resolvedWithPnl.filter(t => t.pnl! < 0);
  const worstTrade = lostTrades.length > 0 ? lostTrades.reduce((w, t) => t.pnl! < w.pnl! ? t : w) : null;

  const wonPnl  = trades.filter(isWon).reduce((s, t) => s + (t.pnl ?? 0), 0);
  const lostPnl = trades.filter(isLost).reduce((s, t) => s + (t.pnl ?? 0), 0);

  const buyHomeTrades  = trades.filter(t => t.action !== "BUY_AWAY" && (t.status === "WON" || t.status === "LOST" || t.status === "CLOSED"));
  const buyHomeWins    = buyHomeTrades.filter(isWon).length;
  const buyHomeWinRate = buyHomeTrades.length > 0 ? (buyHomeWins / buyHomeTrades.length) * 100 : null;
  const buyAwayTrades  = trades.filter(t => t.action === "BUY_AWAY" && (t.status === "WON" || t.status === "LOST" || t.status === "CLOSED"));
  const buyAwayWins    = buyAwayTrades.filter(isWon).length;
  const buyAwayWinRate = buyAwayTrades.length > 0 ? (buyAwayWins / buyAwayTrades.length) * 100 : null;

  const teamCounts       = trades.reduce((acc, t) => { acc[t.target_team] = (acc[t.target_team] ?? 0) + 1; return acc; }, {} as Record<string, number>);
  const mostTargetedTeam = Object.entries(teamCounts).sort((a, b) => b[1] - a[1])[0] ?? null;

  const lastTrade = trades.length > 0 ? trades[0] : null;

  const sparkData = (() => {
    let running = 0;
    return [...trades].reverse().map(t => { running += t.pnl ?? 0; return { pnl: +running.toFixed(2) }; });
  })();

  const gamesWithMarket = liveGames.filter(g => g.market_odds?.polymarket_prob != null);
  const avgMarketEdge = gamesWithMarket.length > 0
    ? gamesWithMarket.reduce((s, g) => s + Math.abs(g.market_odds.market_edge ?? 0), 0) / gamesWithMarket.length * 100
    : null;

  const totalSignals   = liveGames.reduce((s, g) => s + g.signal_count, 0);
  const gamesInPlay    = liveGames.filter(g => g.period >= 1 && g.period <= 4).length;
  const bestLiveGame   = liveGames.length > 0
    ? liveGames.reduce((best, g) => {
        const ea = Math.abs(g.market_odds?.market_edge ?? g.predictions.edge);
        const eb = Math.abs(best.market_odds?.market_edge ?? best.predictions.edge);
        return ea > eb ? g : best;
      })
    : null;
  const bestLiveEdgePct = bestLiveGame
    ? Math.abs(bestLiveGame.market_odds?.market_edge ?? bestLiveGame.predictions.edge) * 100
    : 0;

  const openTrades = trades.filter(t => t.status === "OPEN");

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 font-mono">

      {/* ── Sticky header ── */}
      <div className="sticky top-0 z-10 border-b border-gray-800 bg-gray-950/90 backdrop-blur">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-green-400 tracking-widest">COURTSIDE ALPHA</h1>
            <p className="text-xs text-gray-600 mt-0.5">Quantitative prediction market engine</p>
          </div>
          <div className="flex items-center gap-4 text-xs">
            {lastUpdated && (
              <span className="text-gray-600 hidden sm:block">
                Updated {lastUpdated.toLocaleTimeString()}
              </span>
            )}
            <span className={`flex items-center gap-1.5 font-bold ${connected ? "text-green-400" : "text-red-400"}`}>
              <span className={`w-2 h-2 rounded-full ${connected ? "bg-green-400 animate-pulse" : "bg-red-500"}`} />
              {connected ? "LIVE" : "OFFLINE"}
            </span>
          </div>
        </div>
      </div>

      {/* ── Tab bar ── */}
      <div className="border-b border-gray-800 bg-gray-950">
        <div className="max-w-7xl mx-auto px-6">
          <nav className="flex gap-1 -mb-px">
            {(
              [
                { id: "overview",   label: "Overview" },
                { id: "live",       label: "Live Games",       badge: liveGames.length > 0 ? liveGames.length : undefined },
                { id: "positions",  label: "Active Positions", badge: open > 0 ? open : undefined },
                { id: "history",    label: "Trade History",    badge: trades.length > 0 ? trades.length : undefined },
                { id: "analytics",  label: "Analytics" },
              ] as { id: Tab; label: string; badge?: number }[]
            ).map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-3 text-xs font-bold uppercase tracking-widest border-b-2 transition-colors flex items-center gap-2 ${
                  activeTab === tab.id
                    ? "border-green-400 text-green-400"
                    : "border-transparent text-gray-500 hover:text-gray-300 hover:border-gray-700"
                }`}
              >
                {tab.label}
                {tab.badge != null && (
                  <span className="bg-yellow-500/20 text-yellow-400 border border-yellow-700/40 rounded-full px-1.5 py-px text-xs leading-none">
                    {tab.badge}
                  </span>
                )}
              </button>
            ))}
          </nav>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8 space-y-10">

        {/* ── Overview tab ── */}
        {activeTab === "overview" && (
          <>
            {wallet && (
              <section>
                <SectionHeader label="Testnet Wallet" color="text-cyan-400" count={-1} />
                <div className="bg-gray-900 border border-cyan-900/40 rounded-lg p-5 space-y-3">
                  <div className="flex flex-wrap items-center gap-x-8 gap-y-2 text-xs">
                    <DataPoint label="Address">
                      <span className="text-cyan-300 font-semibold">{wallet.address}</span>
                    </DataPoint>
                    <DataPoint label="Chain">
                      <span className="text-gray-400">{wallet.chain} (ID {wallet.chain_id})</span>
                    </DataPoint>
                    <DataPoint label="Signed orders">
                      <span className="text-purple-400 font-semibold">{signedCount}</span>
                    </DataPoint>
                  </div>
                  <div className="flex items-end gap-6 pt-1">
                    <div>
                      <p className="text-gray-500 text-xs uppercase tracking-widest mb-1">Fake USDC Balance</p>
                      <p className={`text-3xl font-bold ${wallet.usdc_balance >= wallet.initial_usdc ? "text-green-400" : wallet.usdc_balance >= wallet.initial_usdc * 0.8 ? "text-yellow-400" : "text-red-400"}`}>
                        ${wallet.usdc_balance.toFixed(2)}
                      </p>
                      <p className="text-gray-600 text-xs mt-0.5">
                        started at ${wallet.initial_usdc.toLocaleString()} &middot;{" "}
                        <span className={wallet.usdc_balance >= wallet.initial_usdc ? "text-green-500" : "text-red-500"}>
                          {wallet.usdc_balance >= wallet.initial_usdc ? "+" : ""}
                          ${(wallet.usdc_balance - wallet.initial_usdc).toFixed(2)} net
                        </span>
                      </p>
                    </div>
                    <div className="h-12 w-px bg-gray-800" />
                    <div>
                      <p className="text-gray-500 text-xs uppercase tracking-widest mb-1">EIP-712 Signing</p>
                      <p className="text-xs text-gray-300">
                        Polymarket CLOB orders are signed with a secp256k1 test key.<br />
                        Each order hash is stored in the DB &mdash; ready for CLOB submission.
                      </p>
                    </div>
                  </div>
                </div>
              </section>
            )}

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard label="Simulated PnL" value={`${totalPnl >= 0 ? "+" : ""}$${totalPnl.toFixed(2)}`} color={totalPnl >= 0 ? "text-green-400" : "text-red-400"} sub={`$${totalStaked.toFixed(0)} total staked`} />
              <StatCard label="ROI" value={roi != null ? `${roi >= 0 ? "+" : ""}${roi.toFixed(2)}%` : "\u2014"} color={roi != null ? (roi >= 0 ? "text-green-400" : "text-red-400") : "text-gray-500"} sub={`on $${totalStaked.toFixed(0)} staked`} />
              <StatCard label="Win Rate" value={`${winRate.toFixed(1)}%`} color="text-blue-400" sub={resolved > 0 ? `${won}W / ${lost}L` : "No resolved bets yet"} />
              <StatCard label="Avg Stake" value={avgStake != null ? `$${avgStake.toFixed(0)}` : "\u2014"} color="text-gray-300" sub={`${trades.length} trades total`} />
              <StatCard label="Open Positions" value={String(open)} color="text-yellow-400" sub={open > 0 ? "awaiting resolution" : "none active"} />
              <StatCard label="Live Games" value={String(liveGames.length)} color="text-cyan-400" sub={gamesWithMarket.length > 0 ? `${gamesWithMarket.length} with market odds` : "no market data"} />
              <StatCard label="Avg Edge" value={avgEdge != null ? `${avgEdge.toFixed(1)}%` : "\u2014"} color="text-purple-400" sub="model vs market (trades)" />
              <StatCard label="Market Edge" value={avgMarketEdge != null ? `${avgMarketEdge.toFixed(1)}%` : "\u2014"} color="text-orange-400" sub="live model vs Polymarket" />
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
              <div className="xl:col-span-2 bg-gray-900 border border-gray-800 rounded-lg p-5">
                <p className="text-xs text-gray-500 uppercase tracking-widest mb-1">Cumulative PnL</p>
                <p className="text-xs text-gray-600 mb-4">Running total profit/loss across all trades in chronological order.</p>
                <SparklineChart data={sparkData} />
              </div>
              <div className="space-y-4">
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <p className="text-xs text-gray-500 uppercase tracking-widest mb-3">Trade Breakdown</p>
                  {trades.length > 0 ? (
                    <>
                      <div className="flex gap-0.5 h-2.5 rounded-full overflow-hidden mb-2">
                        {won  > 0 && <div className="bg-green-500"  style={{ flex: won  }} />}
                        {lost > 0 && <div className="bg-red-500"    style={{ flex: lost }} />}
                        {open > 0 && <div className="bg-yellow-500" style={{ flex: open }} />}
                      </div>
                      <div className="flex gap-4 text-xs">
                        <span className="text-green-400">{won} WON</span>
                        <span className="text-red-400">{lost} LOST</span>
                        <span className="text-yellow-400">{open} OPEN</span>
                      </div>
                      <div className="mt-3 pt-3 border-t border-gray-800 grid grid-cols-2 gap-2">
                        <DataPoint label="Won PnL"><span className="text-green-400 font-bold">+${wonPnl.toFixed(2)}</span></DataPoint>
                        <DataPoint label="Lost PnL"><span className="text-red-400 font-bold">${lostPnl.toFixed(2)}</span></DataPoint>
                      </div>
                    </>
                  ) : (
                    <p className="text-gray-600 text-xs">No trades yet</p>
                  )}
                </div>
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                  <p className="text-xs text-gray-500 uppercase tracking-widest mb-2">Current Streak</p>
                  {streakType ? (
                    <p className={`text-3xl font-bold ${streakType === "W" ? "text-green-400" : "text-red-400"}`}>
                      {streakType}{streakCount}
                    </p>
                  ) : (
                    <p className="text-gray-600 text-sm">No resolved trades yet</p>
                  )}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className={`bg-gray-900 rounded-lg p-4 ${bestTrade ? "border border-green-900/30" : "border border-gray-800"}`}>
                <p className="text-xs text-gray-500 uppercase tracking-widest mb-2">Best Trade</p>
                {bestTrade ? (<><p className="text-green-400 font-bold text-xl">+${bestTrade.pnl!.toFixed(2)}</p><p className="text-gray-400 text-xs mt-1 truncate" title={bestTrade.target_team}>{bestTrade.target_team}</p></>) : <p className="text-gray-600 text-sm">&mdash;</p>}
              </div>
              <div className={`bg-gray-900 rounded-lg p-4 ${worstTrade ? "border border-red-900/30" : "border border-gray-800"}`}>
                <p className="text-xs text-gray-500 uppercase tracking-widest mb-2">Worst Trade</p>
                {worstTrade ? (<><p className="text-red-400 font-bold text-xl">${worstTrade.pnl!.toFixed(2)}</p><p className="text-gray-400 text-xs mt-1 truncate" title={worstTrade.target_team}>{worstTrade.target_team}</p></>) : <p className="text-gray-600 text-sm">&mdash;</p>}
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <p className="text-xs text-gray-500 uppercase tracking-widest mb-2">Most Traded</p>
                {mostTargetedTeam ? (<><p className="text-gray-200 font-bold text-sm truncate" title={mostTargetedTeam[0]}>{mostTargetedTeam[0]}</p><p className="text-gray-600 text-xs mt-1">{mostTargetedTeam[1]} trades</p></>) : <p className="text-gray-600 text-sm">&mdash;</p>}
              </div>
              <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
                <p className="text-xs text-gray-500 uppercase tracking-widest mb-2">Last Trade</p>
                {lastTrade ? (<><p className="text-gray-200 font-bold">{timeSince(lastTrade.timestamp)}</p><p className="text-gray-600 text-xs mt-1 truncate" title={lastTrade.target_team}>{lastTrade.target_team}</p></>) : <p className="text-gray-600 text-sm">&mdash;</p>}
              </div>
            </div>

            {(buyHomeWinRate != null || buyAwayWinRate != null) && (
              <section>
                <SectionHeader label="Win Rate by Action" count={-1} color="text-blue-400" />
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <ActionWinRateCard action="BUY_HOME" winRate={buyHomeWinRate} count={buyHomeTrades.length} wins={buyHomeWins} />
                  <ActionWinRateCard action="BUY_AWAY" winRate={buyAwayWinRate} count={buyAwayTrades.length} wins={buyAwayWins} />
                </div>
              </section>
            )}

            <section>
              <SectionHeader label="System Pipeline Feed" count={-1} pulse color="text-green-400" />
              <LivePipelineFeed events={feedEvents} />
            </section>

            {liveGames.length > 0 && (
              <section>
                <SectionHeader label="Live Signal Summary" count={-1} pulse color="text-cyan-400" />
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  <StatCard label="Active Signals" value={String(totalSignals)} color={totalSignals > 0 ? "text-yellow-400" : "text-gray-500"} sub="across all live games" />
                  <StatCard label="Games In-Play" value={String(gamesInPlay)} color="text-cyan-400" sub={`of ${liveGames.length} tracked`} />
                  <StatCard label="Best Live Edge" value={bestLiveEdgePct > 0 ? `${bestLiveEdgePct.toFixed(1)}%` : "\u2014"} color={bestLiveEdgePct >= 5 ? "text-green-400" : "text-gray-400"} sub={bestLiveGame ? `${bestLiveGame.home_team} vs ${bestLiveGame.away_team}` : "\u2014"} />
                  <StatCard label="Market Data" value={`${gamesWithMarket.length}/${liveGames.length}`} color="text-orange-400" sub="games with Polymarket odds" />
                </div>
                {bestLiveGame && bestLiveEdgePct >= 5 && (
                  <div className="bg-green-950/30 border border-green-800/40 rounded-lg p-4 flex items-center justify-between">
                    <div>
                      <p className="text-xs text-green-600 uppercase tracking-widest mb-1">Signal Detected</p>
                      <p className="text-green-300 font-bold">{bestLiveGame.home_team} vs {bestLiveGame.away_team}</p>
                      <p className="text-gray-400 text-xs mt-0.5">Q{bestLiveGame.period} &middot; {bestLiveGame.score.home}&ndash;{bestLiveGame.score.away}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-green-400 font-bold text-2xl">{bestLiveEdgePct.toFixed(1)}%</p>
                      <p className="text-gray-500 text-xs">edge</p>
                    </div>
                  </div>
                )}
              </section>
            )}
          </>
        )}

        {/* ── Live Games tab ── */}
        {activeTab === "live" && (
          <section>
            <SectionHeader label="Live Games \u2014 Model vs Market" count={liveGames.length} pulse color="text-cyan-400" />
            {liveGames.length === 0 ? (
              <div className="text-gray-600 text-sm py-12 text-center border border-gray-800 rounded-lg">
                No live games detected &mdash; waiting for active NBA games&hellip;
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                {liveGames.map(g => <LiveGameCard key={g.game_id} game={g} />)}
              </div>
            )}
          </section>
        )}

        {/* ── Active Positions tab ── */}
        {activeTab === "positions" && (
          <section>
            <SectionHeader label="Active Positions" count={openTrades.length} pulse color="text-yellow-400" />
            {loading ? (
              <div className="text-gray-600 text-sm py-6">Loading&hellip;</div>
            ) : openTrades.length === 0 ? (
              <div className="text-gray-600 text-sm py-12 text-center border border-gray-800 rounded-lg">
                No open positions &mdash; waiting for edge signal&hellip;
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                {openTrades.map(t => <ActivePositionCard key={t.id} trade={t} />)}
              </div>
            )}
          </section>
        )}

        {/* ── Trade History tab ── */}
        {activeTab === "history" && (
          <section>
            <SectionHeader label="Trade History" count={trades.length} color="text-gray-400" />
            {loading ? (
              <div className="text-gray-600 text-sm py-12 text-center">Loading&hellip;</div>
            ) : trades.length === 0 ? (
              <div className="text-gray-600 text-sm py-12 text-center border border-gray-800 rounded-lg">
                No trades logged yet &mdash; waiting for edge signal&hellip;
              </div>
            ) : (
              <div className="overflow-x-auto rounded-lg border border-gray-800">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-gray-900 text-gray-500 text-left uppercase tracking-wider">
                      <th className="px-4 py-3">Time</th>
                      <th className="px-4 py-3">Game / Team</th>
                      <th className="px-4 py-3">Action</th>
                      <th className="px-4 py-3">Stake</th>
                      <th className="px-4 py-3"><div>Mkt Win %</div><div className="text-gray-600 normal-case font-normal text-xs tracking-normal">P(bet team wins)</div></th>
                      <th className="px-4 py-3"><div>Mdl Win %</div><div className="text-gray-600 normal-case font-normal text-xs tracking-normal">P(bet team wins)</div></th>
                      <th className="px-4 py-3">Edge</th>
                      <th className="px-4 py-3">Order Hash</th>
                      <th className="px-4 py-3">Status</th>
                      <th className="px-4 py-3 text-right">PnL</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trades.map(t => {
                      const { mkt, mdl, edge } = alignedProbs(t);
                      const edgePct = edge * 100;
                      return (
                        <tr key={t.id} className="border-t border-gray-800/60 hover:bg-gray-900/50 transition-colors">
                          <td className="px-4 py-3 text-gray-500 whitespace-nowrap">{fmt(t.timestamp)}</td>
                          <td className="px-4 py-3 max-w-xs text-gray-300" title={t.target_team}>
                            <div className="truncate">{t.target_team}</div>
                            {t.bought_home !== null && (
                              <div className="flex items-center gap-1 mt-0.5">
                                <span className={`text-xs font-medium ${t.bought_home ? "text-blue-400" : "text-violet-400"}`}>
                                  {t.bought_home ? "\u25b2 HOME" : "\u25bd AWAY"}
                                </span>
                                <span className="text-xs text-gray-600">&middot; {betTeam(t)}</span>
                              </div>
                            )}
                          </td>
                          <td className="px-4 py-3">
                            <span className={`font-semibold ${t.action.startsWith("BUY") ? "text-green-400" : "text-red-400"}`}>{t.action}</span>
                          </td>
                          <td className="px-4 py-3 text-gray-400">${t.stake_amount.toFixed(0)}</td>
                          <td className="px-4 py-3"><div className="text-gray-300">{(mkt * 100).toFixed(1)}%</div><div className="text-xs text-gray-600 mt-0.5">{betTeam(t)}</div></td>
                          <td className="px-4 py-3"><div className="text-gray-300">{(mdl * 100).toFixed(1)}%</div><div className="text-xs text-gray-600 mt-0.5">{betTeam(t)}</div></td>
                          <td className={`px-4 py-3 font-bold ${edgePct >= 0 ? "text-green-400" : "text-red-400"}`}>{edgePct >= 0 ? "+" : ""}{edgePct.toFixed(1)}%</td>
                          <td className="px-4 py-3">
                            {t.order_hash ? (
                              <span className="text-cyan-500 cursor-help font-mono" title={t.order_hash}>{shortHash(t.order_hash, 12)}</span>
                            ) : (
                              <span className="text-gray-700">&mdash;</span>
                            )}
                          </td>
                          <td className="px-4 py-3"><StatusBadge status={t.status} /></td>
                          <td className={`px-4 py-3 text-right font-semibold ${(t.pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400"}`}>
                            {t.pnl != null ? `${t.pnl >= 0 ? "+" : ""}$${t.pnl.toFixed(2)}` : "\u2014"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        )}

        {/* ── Analytics tab ── */}
        {activeTab === "analytics" && (
          <section>
            <SectionHeader label="Model vs Market \u2014 Confidence Intervals" count={-1} color="text-purple-400" />
            {trades.length < 2 ? (
              <div className="text-gray-600 text-sm py-12 text-center border border-gray-800 rounded-lg">
                Need at least 2 trades to display charts&hellip;
              </div>
            ) : (
              <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                <div className="xl:col-span-2 bg-gray-900 border border-gray-800 rounded-lg p-5">
                  <p className="text-xs text-gray-500 mb-1 uppercase tracking-widest">Probability Over Trades</p>
                  <p className="text-xs text-gray-600 mb-4">Purple band = model &plusmn;5% CI (edge threshold). Trades trigger when market exits this band.</p>
                  <ProbabilityChart trades={[...trades].slice(0, 30).reverse()} />
                </div>
                <div className="bg-gray-900 border border-gray-800 rounded-lg p-5">
                  <p className="text-xs text-gray-500 mb-1 uppercase tracking-widest">Edge Distribution</p>
                  <p className="text-xs text-gray-600 mb-4">Model edge per trade. Green = positive (model &gt; market).</p>
                  <EdgeChart trades={[...trades].slice(0, 20).reverse()} />
                </div>
              </div>
            )}
          </section>
        )}

      </div>

      <ToastStack toasts={toasts} onDismiss={(id) => setToasts(prev => prev.filter(t => t.id !== id))} />
    </main>
  );
}
