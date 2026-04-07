export interface Trade {
  id:                  string;
  timestamp:           string;
  game_id:             string;
  target_team:         string;
  action:              string;
  market_implied_prob: number;
  model_implied_prob:  number;
  stake_amount:        number;
  status:              string;
  pnl:                 number | null;
  order_hash:          string | null;
  signed_tx:           string | null;
  bought_home:         boolean | null;
}

export interface WalletInfo {
  address:      string;
  usdc_balance: number;
  chain_id:     number;
  chain:        string;
  initial_usdc: number;
}

export interface LiveGame {
  game_id:      string;
  home_team:    string;
  away_team:    string;
  home_team_id: number;
  away_team_id: number;
  score:        { home: number; away: number };
  period:       number;
  game_clock:   string;
  predictions: {
    win_probability:   number;
    proxy_probability: number;
    predicted_margin:  number;
    edge:              number;
    abs_edge:          number;
    edge_confidence:   number;
    kelly_size:        number;
  };
  market_odds: {
    polymarket_prob:  number | null;
    market_edge:      number | null;
    market_abs_edge:  number | null;
    source:           string | null;
    volume:           number | null;
    spread:           number | null;
    total:            number | null;
  };
  signals:      unknown[];
  signal_count: number;
}

export type Tab = "overview" | "live" | "positions" | "history" | "analytics";

export interface FeedEvent {
  id:        string;
  type:      "NBA_DATA" | "ML_PREDICTION" | "TRADE";
  message:   string;
  detail:    string;
  timestamp: Date;
}

export interface ToastNotification {
  id:    string;
  trade: Trade;
}
