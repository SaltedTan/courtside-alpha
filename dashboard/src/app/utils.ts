import type { Trade } from "./types";

export function fmt(dt: string) {
  return new Date(dt).toLocaleString(undefined, {
    month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

export function shortHash(h: string | null, len = 10): string {
  if (!h) return "\u2014";
  return h.length > len + 4 ? `${h.slice(0, len)}\u2026` : h;
}

/** Return market/model probs aligned to the traded team's perspective.
 *  Raw DB values are always P(home wins). When bought_home=false we flip both. */
export function alignedProbs(t: Trade): { mkt: number; mdl: number; edge: number } {
  const flip = t.bought_home === false;
  const mkt = flip ? 1 - t.market_implied_prob : t.market_implied_prob;
  const mdl = flip ? 1 - t.model_implied_prob  : t.model_implied_prob;
  return { mkt, mdl, edge: mdl - mkt };
}

/** The team whose win-probability the market%/model% columns are showing.
 *  For BUY rows the action encodes the team directly. For SELL rows use target_team. */
export function betTeam(t: Trade): string {
  if (t.action.startsWith("BUY_")) {
    return t.action.slice(4).replace(/_/g, " ")
      .toLowerCase().replace(/\b\w/g, c => c.toUpperCase());
  }
  return t.target_team;
}

export function timeSince(dateStr: string): string {
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}
