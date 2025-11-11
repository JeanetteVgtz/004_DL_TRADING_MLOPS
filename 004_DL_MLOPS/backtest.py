import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class Position:
    side: str
    n_shares: float
    entry_price: float
    sl: float
    tp: float

def run_backtest(df, stop_loss=0.02, take_profit=0.04, n_shares=1,
                 com=0.125/100, borrow_rate=0.25/100,
                 price_col="close", initial_cash=1_000_000):

    df = df.copy()

    cash = float(initial_cash)
    active_long = []
    active_short = []
    portfolio_values = []
    trade_pnls = []

    FEE_LONG = com
    FEE_SHORT = com + borrow_rate


    for i, row in df.iterrows():
        price = float(row[price_col])
        signal = int(row["signal"])  # 1=buy, 0=hold, -1=sell

        pnl_this_step = 0
        closed_any = False

        # ---- CLOSE LONGS ----
        for pos in active_long.copy():
            pnl_realized = 0
            if price >= pos.tp or price <= pos.sl:
                # Realize PnL
                entry_fee = pos.entry_price * pos.n_shares * FEE_LONG
                exit_fee  = price * pos.n_shares * FEE_LONG
                pnl_realized = (price - pos.entry_price) * pos.n_shares - entry_fee - exit_fee

                # Flujo de caja al cerrar long: cobras venta neta de comisión
                cash += price * pos.n_shares * (1 - FEE_LONG)
                active_long.remove(pos)

                pnl_this_step += pnl_realized
                closed_any = True

        # ---- CLOSE SHORTS ----
        for pos in active_short.copy():
            pnl_realized = 0
            if price <= pos.tp or price >= pos.sl:
                # PnL for short: (entry_price - exit_price) * n_shares
                pnl_gross = (pos.entry_price - price) * pos.n_shares
                # Fees de entrada y salida (simplificado)
                entry_fee = pos.entry_price * pos.n_shares * FEE_SHORT
                exit_fee  = price * pos.n_shares * FEE_SHORT
                pnl_realized = pnl_gross - entry_fee - exit_fee
                
                cash += (pnl_gross * (1 - FEE_SHORT)) + (pos.entry_price * pos.n_shares)

                active_short.remove(pos)
                pnl_this_step += pnl_realized
                closed_any = True

        # ---- OPEN LONG ----
        if signal == 1 and len(active_short) == 0:
            n_shares_dynamic = max(1, (cash * 0.02) / price)
            cost = price * n_shares_dynamic * (1 + FEE_LONG)
            if cash > cost:
                cash -= cost
                pos = Position("long", n_shares_dynamic, price,
                               price * (1 - stop_loss),
                               price * (1 + take_profit))
                active_long.append(pos)

        # ---- OPEN SHORT ----
        if signal == -1 and len(active_long) == 0:
            n_shares_dynamic = max(1, (cash * 0.02) / price)
            cost = price * n_shares_dynamic * (1 + FEE_SHORT)
            if cash > cost:
                cash -= cost
                pos = Position("short", n_shares_dynamic, price,
                               price * (1 + stop_loss),   # SL arriba
                               price * (1 - take_profit)) # TP abajo
                active_short.append(pos)

        # ---- PORTFOLIO VALUE ----
        # Cash + value of active positions
        value_longs = sum(p.n_shares * price for p in active_long)
        value_shorts = sum(
            (p.entry_price - price) * p.n_shares + (p.entry_price * p.n_shares)
            for p in active_short
        )
        equity = cash + value_longs + value_shorts
        portfolio_values.append(equity)

        trade_pnls.append(pnl_this_step if closed_any else 0)

    # Force close all positions at the end
    if len(df) > 0:
        last_price = float(df.iloc[-1][price_col])

        if active_long:
            cash += last_price * sum(p.n_shares for p in active_long) * (1 - FEE_LONG)
            active_long.clear()

        if active_short:
            for p in active_short:
                pnl = (p.entry_price - last_price) * p.n_shares
                cash += (pnl * (1 - FEE_SHORT)) + (p.entry_price * p.n_shares)
            active_short.clear()


        portfolio_values[-1] = cash


    df["portfolio_value"] = portfolio_values
    df["trade_pnl"] = trade_pnls

    return df, cash



