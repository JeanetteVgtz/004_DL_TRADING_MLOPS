# backtest.py
"""
Backtesting engine for CNN-generated trading signals.
Simulates portfolio evolution with SL/TP, commissions, and both long & short positions.
"""

import pandas as pd
from dataclasses import dataclass


@dataclass
class Trade:
    side: str       # "long" or "short"
    qty: float      # fixed position size (number of shares)
    entry: float    # entry price
    sl: float       # stop-loss price
    tp: float       # take-profit price


def execute_backtest(
    data: pd.DataFrame,
    stop_thr: float = 0.02,         # 2% stop loss
    tp_thr: float = 0.04,           # 4% take profit
    lot_size: float = 1.0,          # number of shares per trade
    commission: float = 0.125 / 100,# 0.125% per leg
    col_price: str = "close",
    start_cap: float = 1_000_000,   # initial capital
):
    """
    Execute a rule-based backtest using trading signals.

    Logic:
        - Initial cash balance
        - Open/close LONG or SHORT based on signal column (1=long, 0=hold, -1=short)
        - SL/TP control and commission applied on both entry and exit
        - Portfolio value = cash + marked-to-market open positions
        - Returns dataframe with 'portfolio_value' and 'trade_pnl', plus final cash balance
    """
    df = data.copy()

    cash = float(start_cap)
    active_long: list[Trade] = []
    active_short: list[Trade] = []
    portfolio_values: list[float] = []
    trade_pnls: list[float] = []

    print("=" * 70)
    print("STEP 5: BACKTEST EXECUTION")
    print("=" * 70)
    print(f"Starting capital: {start_cap:,.2f}\n")

    for _, row in df.iterrows():
        price = float(row[col_price])
        signal = int(row["signal"])  # signal must be in {-1, 0, 1}

        pnl_this_step = 0.0
        closed_any = False

        # =============================
        # CLOSE POSITIONS (SL / TP)
        # =============================
        # LONGS
        for pos in active_long.copy():
            if price >= pos.tp or price <= pos.sl:
                entry_fee = pos.entry * pos.qty * commission
                exit_fee = price * pos.qty * commission
                pnl_realized = (price - pos.entry) * pos.qty - entry_fee - exit_fee

                cash += price * pos.qty * (1 - commission)
                active_long.remove(pos)

                pnl_this_step += pnl_realized
                closed_any = True

        # SHORTS
        for pos in active_short.copy():
            if price <= pos.tp or price >= pos.sl:
                pnl_gross = (pos.entry - price) * pos.qty
                entry_fee = pos.entry * pos.qty * commission
                exit_fee = price * pos.qty * commission
                pnl_realized = pnl_gross - entry_fee - exit_fee

                cash += (pnl_gross * (1 - commission)) + (pos.entry * pos.qty)
                active_short.remove(pos)

                pnl_this_step += pnl_realized
                closed_any = True

        # =============================
        # OPEN POSITIONS (based on signal)
        # =============================
        if signal == 1:  # LONG
            cost = price * lot_size * (1 + commission)
            if cash >= cost:
                cash -= cost
                active_long.append(
                    Trade(
                        side="long",
                        qty=float(lot_size),
                        entry=price,
                        sl=price * (1 - stop_thr),
                        tp=price * (1 + tp_thr),
                    )
                )

        elif signal == -1:  # SHORT
            cost = price * lot_size * (1 + commission)
            if cash >= cost:
                cash -= cost
                active_short.append(
                    Trade(
                        side="short",
                        qty=float(lot_size),
                        entry=price,
                        sl=price * (1 + stop_thr),  # stop above
                        tp=price * (1 - tp_thr),    # take profit below
                    )
                )

        # =============================
        # PORTFOLIO VALUATION
        # =============================
        val = cash
        for pos in active_long:
            val += pos.qty * price
        for pos in active_short:
            val += (pos.entry - price) * pos.qty + (pos.entry * pos.qty)

        portfolio_values.append(val)
        trade_pnls.append(pnl_this_step if closed_any else 0.0)

    # =============================
    # FINAL CLOSURE
    # =============================
    if len(df) > 0:
        last_price = float(df.iloc[-1][col_price])

        # Close all remaining LONG positions
        if active_long:
            total_qty = sum(p.qty for p in active_long)
            cash += last_price * total_qty * (1 - commission)
            active_long.clear()

        # Close all remaining SHORT positions
        if active_short:
            for p in active_short:
                pnl_gross = (p.entry - last_price) * p.qty
                cash += (pnl_gross * (1 - commission)) + (p.entry * p.qty)
            active_short.clear()

        portfolio_values[-1] = cash  # last portfolio value = final cash

    df["portfolio_value"] = portfolio_values
    df["trade_pnl"] = trade_pnls

    print(f"\n✅ Backtest completed successfully.")
    print(f"Final capital: {cash:,.2f}")
    print(f"Net return: {(cash / start_cap - 1) * 100:.2f}%")
    print("=" * 70)

    return df, cash


if __name__ == "__main__":
    # Example usage
    print("\nRunning standalone backtest example...")
    try:
        df = pd.read_csv("data/processed/val_features.csv")
        # Dummy signal for testing
        df["signal"] = 0
        df.loc[::50, "signal"] = 1   # open long every 50 bars
        df.loc[25::50, "signal"] = -1 # open short every 50 bars

        results, final_cap = execute_backtest(df)
    except Exception as e:
        print(f"⚠️ Error during backtest: {e}")




