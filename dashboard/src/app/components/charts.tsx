"use client";

import {
  ComposedChart, Area, Line, XAxis, YAxis,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
  BarChart, Bar, LineChart,
} from "recharts";
import type { Trade } from "../types";

function ProbTooltip({ active, payload, label }: {
  active?: boolean; payload?: Array<{ dataKey: string; value: number }>; label?: string;
}) {
  if (!active || !payload?.length) return null;
  const model  = payload.find(p => p.dataKey === "model");
  const market = payload.find(p => p.dataKey === "market");
  const edge   = model && market ? model.value - market.value : null;
  return (
    <div className="bg-gray-950 border border-gray-700 rounded-md px-3 py-2 text-xs space-y-1">
      <p className="text-gray-400 font-semibold">{label}</p>
      {model  && <p className="text-purple-400">Model:  {model.value.toFixed(1)}%</p>}
      {market && <p className="text-yellow-400">Market: {market.value.toFixed(1)}%</p>}
      {edge != null && (
        <p className={`font-bold ${edge >= 0 ? "text-green-400" : "text-red-400"}`}>
          Edge: {edge >= 0 ? "+" : ""}{edge.toFixed(1)}%
        </p>
      )}
    </div>
  );
}

export function ProbabilityChart({ trades }: { trades: Trade[] }) {
  const CI = 5;
  const data = trades.map(t => {
    const modelPct = +(t.model_implied_prob * 100).toFixed(1);
    const low      = +Math.max(0,   modelPct - CI).toFixed(1);
    const high     = +Math.min(100, modelPct + CI).toFixed(1);
    return {
      label:   t.target_team.split(" ").slice(-1)[0],
      market:  +(t.market_implied_prob * 100).toFixed(1),
      model:   modelPct,
      ciBase:  low,
      ciWidth: +(high - low).toFixed(1),
    };
  });

  return (
    <ResponsiveContainer width="100%" height={260}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
        <defs>
          <linearGradient id="ciGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#8b5cf6" stopOpacity={0.30} />
            <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.05} />
          </linearGradient>
        </defs>
        <XAxis dataKey="label" tick={{ fill: "#6b7280", fontSize: 10 }} tickLine={false} axisLine={{ stroke: "#374151" }} />
        <YAxis domain={[0, 100]} tick={{ fill: "#6b7280", fontSize: 10 }} tickLine={false} axisLine={false}
               tickFormatter={v => `${v}%`} width={38} />
        <Tooltip content={<ProbTooltip />} />
        <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "10px" }}
                formatter={v => <span style={{ color: "#9ca3af" }}>{v}</span>} />
        <ReferenceLine y={50} stroke="#374151" strokeDasharray="3 3" />
        <Area type="monotone" dataKey="ciBase"  stackId="ci" stroke="none" fill="transparent" legendType="none" tooltipType="none" />
        <Area type="monotone" dataKey="ciWidth" stackId="ci" stroke="none" fill="url(#ciGrad)" name="Model CI (\u00b15%)" legendType="square" />
        <Line type="monotone" dataKey="model"  name="Model Probability"  stroke="#8b5cf6" strokeWidth={2}
              dot={{ r: 3, fill: "#8b5cf6", strokeWidth: 0 }} activeDot={{ r: 5 }} />
        <Line type="monotone" dataKey="market" name="Market Probability" stroke="#f59e0b" strokeWidth={2}
              strokeDasharray="5 3" dot={{ r: 3, fill: "#f59e0b", strokeWidth: 0 }} activeDot={{ r: 5 }} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

export function EdgeChart({ trades }: { trades: Trade[] }) {
  const data = trades.map(t => ({
    label: t.target_team.split(" ").slice(-1)[0],
    edge:  +((t.model_implied_prob - t.market_implied_prob) * 100).toFixed(1),
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
        <XAxis dataKey="label" tick={{ fill: "#6b7280", fontSize: 10 }} tickLine={false} axisLine={{ stroke: "#374151" }} />
        <YAxis tick={{ fill: "#6b7280", fontSize: 10 }} tickLine={false} axisLine={false}
               tickFormatter={v => `${v}%`} width={38} />
        <Tooltip
          contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: "6px", fontSize: "11px" }}
          labelStyle={{ color: "#94a3b8" }}
          formatter={(v: number) => [`${v >= 0 ? "+" : ""}${v}%`, "Edge"]}
        />
        <ReferenceLine y={0} stroke="#374151" />
        <Bar dataKey="edge" name="Edge" shape={(props: any) => {
          const { x, y, width, height } = props;
          const fill = (props.value ?? 0) >= 0 ? "#4ade80" : "#f87171";
          return <rect x={x} y={y} width={width} height={Math.abs(height)} fill={fill} fillOpacity={0.85} rx={2} />;
        }} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function SparklineChart({ data }: { data: Array<{ pnl: number }> }) {
  if (data.length < 2) {
    return (
      <div className="h-40 flex items-center justify-center text-gray-700 text-xs">
        Not enough data yet
      </div>
    );
  }
  const color = data[data.length - 1].pnl >= 0 ? "#4ade80" : "#f87171";
  return (
    <ResponsiveContainer width="100%" height={160}>
      <LineChart data={data} margin={{ top: 4, right: 8, left: -8, bottom: 0 }}>
        <YAxis tick={{ fill: "#6b7280", fontSize: 10 }} tickLine={false} axisLine={false}
               tickFormatter={v => `$${v}`} width={48} />
        <Tooltip
          contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: "6px", fontSize: "11px" }}
          formatter={(v: number) => [`${v >= 0 ? "+" : ""}$${v.toFixed(2)}`, "Cum. PnL"]}
          labelFormatter={() => ""}
        />
        <ReferenceLine y={0} stroke="#374151" strokeDasharray="3 3" />
        <Line type="monotone" dataKey="pnl" stroke={color} strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="text-gray-500 w-12 text-right">{label}</span>
      <div className="flex-1 bg-gray-800 rounded-full h-3 overflow-hidden">
        <div className={`${color} h-full rounded-full transition-all duration-500`} style={{ width: `${Math.max(2, value * 100)}%` }} />
      </div>
      <span className="text-gray-300 w-12 font-bold">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}
