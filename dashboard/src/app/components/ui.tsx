"use client";

import { useEffect, useRef } from "react";
import type { FeedEvent, ToastNotification } from "../types";

export function SectionHeader({
  label, count, pulse, color,
}: {
  label: string; count: number; pulse?: boolean; color: string;
}) {
  return (
    <h2 className={`text-xs font-bold uppercase tracking-widest mb-4 flex items-center gap-2 ${color}`}>
      {pulse && <span className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse" />}
      {label}
      {count >= 0 && <span className="text-gray-600 font-normal">({count})</span>}
    </h2>
  );
}

export function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    OPEN:   "bg-yellow-900/50 text-yellow-300 border border-yellow-800/50",
    WON:    "bg-green-900/50  text-green-300  border border-green-800/50",
    LOST:   "bg-red-900/50    text-red-300    border border-red-800/50",
    CLOSED: "bg-gray-800/50   text-gray-300   border border-gray-700/50",
  };
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-bold whitespace-nowrap ${styles[status] ?? "bg-gray-700 text-gray-300"}`}>
      {status}
    </span>
  );
}

const FEED_CONFIG = {
  NBA_DATA:      { label: "NBA DATA",   textColor: "text-cyan-400",  dotColor: "bg-cyan-400"  },
  ML_PREDICTION: { label: "ML MODEL",   textColor: "text-amber-400", dotColor: "bg-amber-400" },
  TRADE:         { label: "TRADE",      textColor: "text-green-400", dotColor: "bg-green-400" },
} as const;

export function LivePipelineFeed({ events }: { events: FeedEvent[] }) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = 0;
  }, [events.length]);

  if (events.length === 0) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-lg h-48 flex items-center justify-center">
        <p className="text-gray-600 text-xs">Waiting for system events\u2026</p>
      </div>
    );
  }

  return (
    <div ref={scrollRef} className="bg-gray-900 border border-gray-800 rounded-lg overflow-y-auto h-56 divide-y divide-gray-800/50">
      {events.map((ev, i) => {
        const cfg = FEED_CONFIG[ev.type];
        return (
          <div
            key={ev.id}
            className={`flex items-start gap-3 px-4 py-2.5 text-xs ${i === 0 ? "bg-gray-800/50" : "hover:bg-gray-800/20"} transition-colors`}
          >
            <span className="text-gray-600 whitespace-nowrap pt-0.5 w-24 shrink-0 tabular-nums">
              {ev.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
            </span>
            <span className={`w-2 h-2 rounded-full mt-1 shrink-0 ${cfg.dotColor} ${i === 0 ? "animate-pulse" : ""}`} />
            <span className={`font-bold uppercase tracking-wider shrink-0 w-20 ${cfg.textColor}`}>
              {cfg.label}
            </span>
            <div className="min-w-0 leading-relaxed">
              <span className="text-gray-200">{ev.message}</span>
              {ev.detail && <span className="text-gray-500 ml-2">{ev.detail}</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function ToastStack({
  toasts,
  onDismiss,
}: {
  toasts: ToastNotification[];
  onDismiss: (id: string) => void;
}) {
  if (toasts.length === 0) return null;
  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col gap-3 pointer-events-none">
      {toasts.map(toast => {
        const edge  = (toast.trade.model_implied_prob - toast.trade.market_implied_prob) * 100;
        const isBuy = toast.trade.action.startsWith("BUY");
        return (
          <div
            key={toast.id}
            className="pointer-events-auto bg-gray-900 border border-green-700/60 rounded-lg p-4 w-72 shadow-xl shadow-black/50 animate-slide-in"
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse shrink-0" />
                <span className="text-green-400 font-bold text-xs uppercase tracking-widest">Trade Executed</span>
              </div>
              <button
                onClick={() => onDismiss(toast.id)}
                className="text-gray-600 hover:text-gray-400 text-xs leading-none"
              >
                \u2715
              </button>
            </div>
            <p className="text-gray-100 font-semibold text-sm mt-2 leading-snug truncate" title={toast.trade.target_team}>
              {toast.trade.target_team}
            </p>
            <div className="flex items-center gap-4 mt-2 text-xs">
              <span className={`font-bold ${isBuy ? "text-green-400" : "text-red-400"}`}>{toast.trade.action}</span>
              <span className={`font-bold ${edge >= 0 ? "text-green-400" : "text-red-400"}`}>
                {edge >= 0 ? "+" : ""}{edge.toFixed(1)}% edge
              </span>
              <span className="text-gray-500">${toast.trade.stake_amount}</span>
            </div>
            {toast.trade.order_hash && (
              <p className="text-cyan-700 text-xs mt-2 font-mono truncate" title={toast.trade.order_hash}>
                {toast.trade.order_hash.slice(0, 22)}\u2026
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
