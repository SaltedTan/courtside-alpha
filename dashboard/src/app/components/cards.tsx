"use client";

import type { Trade, LiveGame } from "../types";
import { fmt, shortHash, alignedProbs } from "../utils";
import { StatusBadge } from "./ui";
import { ProbBar } from "./charts";

export function StatCard({
  label, value, color, sub,
}: {
  label: string; value: string; color: string; sub?: string;
}) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-4">
      <p className="text-gray-500 text-xs uppercase tracking-widest mb-1">{label}</p>
      <p className={`text-2xl font-bold ${color}`}>{value}</p>
      {sub && <p className="text-gray-600 text-xs mt-1">{sub}</p>}
    </div>
  );
}

export function DataPoint({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <span className="text-gray-600 block">{label}</span>
      <div className="mt-0.5">{children}</div>
    </div>
  );
}

export function ActivePositionCard({ trade }: { trade: Trade }) {
  const { edge } = alignedProbs(trade);
  const isBuyHome = trade.bought_home !== false;
  return (
    <div className="bg-gray-900 border border-yellow-900/30 rounded-lg p-4 space-y-3 hover:border-yellow-700/40 transition-colors">
      <div className="flex items-start justify-between gap-3">
        <p className="text-gray-200 text-sm font-semibold leading-snug flex-1 min-w-0 truncate" title={trade.target_team}>
          {trade.target_team}
        </p>
        <StatusBadge status={trade.status} />
      </div>

      <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-xs">
        <DataPoint label="Action">
          <span className={`font-bold ${isBuyHome ? "text-green-400" : "text-orange-400"}`}>
            {trade.action}
          </span>
        </DataPoint>
        <DataPoint label="Stake">
          <span className="text-gray-200 font-semibold">${trade.stake_amount.toFixed(0)} USDC</span>
        </DataPoint>
        <DataPoint label="Market odds">
          <span className="text-gray-300">{(alignedProbs(trade).mkt * 100).toFixed(1)}%</span>
        </DataPoint>
        <DataPoint label="Model confidence">
          <span className="text-gray-300">{(alignedProbs(trade).mdl * 100).toFixed(1)}%</span>
        </DataPoint>
      </div>

      {trade.order_hash && (
        <div className="pt-1 border-t border-gray-800 text-xs">
          <span className="text-gray-600">Order hash </span>
          <span className="text-cyan-500 font-mono" title={trade.order_hash}>
            {shortHash(trade.order_hash, 14)}
          </span>
        </div>
      )}

      <div className="flex items-center justify-between pt-2 border-t border-gray-800">
        <span className="text-gray-600 text-xs">{fmt(trade.timestamp)}</span>
        <span className={`text-sm font-bold ${edge >= 0 ? "text-green-400" : "text-red-400"}`}>
          EDGE {edge >= 0 ? "+" : ""}{edge.toFixed(1)}%
        </span>
      </div>
    </div>
  );
}

export function LiveGameCard({ game }: { game: LiveGame }) {
  const modelProb  = game.predictions.win_probability;
  const marketProb = game.market_odds?.polymarket_prob;
  const marketEdge = game.market_odds?.market_edge;
  const proxyProb  = game.predictions.proxy_probability;
  const margin     = game.predictions.predicted_margin;
  const confidence = game.predictions.edge_confidence;
  const kelly      = game.predictions.kelly_size;
  const hasMarket  = marketProb != null;

  const edgeVal    = hasMarket ? (marketEdge ?? 0) : game.predictions.edge;
  const edgePct    = edgeVal * 100;

  return (
    <div className="bg-gray-900 border border-cyan-900/30 rounded-lg p-4 space-y-3 hover:border-cyan-700/40 transition-colors">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-gray-200 font-bold text-sm">{game.home_team}</span>
          <span className="text-gray-600 text-xs">vs</span>
          <span className="text-gray-200 font-bold text-sm">{game.away_team}</span>
        </div>
        <div className="text-right">
          <span className="text-gray-100 font-bold text-lg">{game.score.home}-{game.score.away}</span>
          <span className="text-gray-500 text-xs ml-2">Q{game.period}</span>
        </div>
      </div>

      <div className="space-y-2">
        <ProbBar label={`Model \u00b7 P(${game.home_team} wins)`} value={modelProb} color="bg-green-500" />
        {hasMarket ? (
          <ProbBar label={`Market \u00b7 P(${game.home_team} wins)`} value={marketProb!} color="bg-blue-500" />
        ) : (
          <ProbBar label={`Proxy \u00b7 P(${game.home_team} wins)`} value={proxyProb} color="bg-gray-500" />
        )}
      </div>

      <div className="grid grid-cols-3 gap-x-4 gap-y-1 text-xs">
        <DataPoint label="Edge">
          <span className={`font-bold ${edgePct >= 0 ? "text-green-400" : "text-red-400"}`}>
            {edgePct >= 0 ? "+" : ""}{edgePct.toFixed(1)}%
          </span>
          <span className="text-gray-600 ml-1 text-xs">
            {edgePct >= 0 ? `\u2191${game.home_team}` : `\u2191${game.away_team}`}
          </span>
        </DataPoint>
        <DataPoint label="Confidence">
          <span className={`font-bold ${confidence >= 0.6 ? "text-green-400" : "text-gray-400"}`}>
            {(confidence * 100).toFixed(0)}%
          </span>
        </DataPoint>
        <DataPoint label="Kelly">
          <span className="text-purple-400 font-bold">{(kelly * 100).toFixed(1)}%</span>
        </DataPoint>
        <DataPoint label="Margin">
          <span className="text-gray-300">{margin >= 0 ? "+" : ""}{margin.toFixed(1)}</span>
        </DataPoint>
        <DataPoint label="Signals">
          <span className={game.signal_count > 0 ? "text-yellow-400 font-bold" : "text-gray-600"}>
            {game.signal_count}
          </span>
        </DataPoint>
        {hasMarket && game.market_odds.volume != null && (
          <DataPoint label="Volume">
            <span className="text-gray-400">${(game.market_odds.volume! / 1000).toFixed(0)}k</span>
          </DataPoint>
        )}
      </div>

      <div className="flex items-center justify-between pt-2 border-t border-gray-800 text-xs">
        <span className="text-gray-600">
          {hasMarket ? `via ${game.market_odds.source}` : "proxy model baseline"}
        </span>
        <span className={`font-bold text-sm ${Math.abs(edgePct) >= 5 ? (edgePct >= 0 ? "text-green-400" : "text-red-400") : "text-gray-500"}`}>
          {Math.abs(edgePct) >= 5 ? "SIGNAL" : "NO EDGE"}
        </span>
      </div>
    </div>
  );
}

export function ActionWinRateCard({ action, winRate, count, wins }: {
  action: string; winRate: number | null; count: number; wins: number;
}) {
  const isHome = action !== "BUY_AWAY";
  return (
    <div className={`bg-gray-900 border rounded-lg p-4 ${isHome ? "border-green-900/30" : "border-orange-900/30"}`}>
      <div className="flex items-center justify-between mb-3">
        <span className={`text-xs font-bold uppercase tracking-widest ${isHome ? "text-green-400" : "text-orange-400"}`}>
          {action}
        </span>
        <span className="text-gray-500 text-xs">{count} resolved</span>
      </div>
      {winRate != null ? (
        <>
          <p className={`text-3xl font-bold ${isHome ? "text-green-400" : "text-orange-400"}`}>
            {winRate.toFixed(1)}%
          </p>
          <p className="text-gray-600 text-xs mt-1">{wins}W / {count - wins}L</p>
          <div className="mt-3 bg-gray-800 rounded-full h-2 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${isHome ? "bg-green-500" : "bg-orange-500"}`}
              style={{ width: `${winRate}%` }}
            />
          </div>
        </>
      ) : (
        <p className="text-gray-600 text-sm">No resolved trades</p>
      )}
    </div>
  );
}
