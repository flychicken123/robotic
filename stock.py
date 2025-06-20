# === STRATEGY PARAMETERS ===
FINAL_STRATEGY_PARAMS = {
    'short_ma': 20,
    'long_ma': 50,
    'atr_trailing_stop_mult': 2.5,
    'swing_low_window': 10 # Window on each side to define a swing low
}

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ta
import datetime

# === PARAMETERS ===
SWING_WINDOW = 3  # Number of bars on each side to define a swing low (was 5)

# NASDAQ 100 tickers (as of June 2024, can be updated)
NASDAQ_100 = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'COST',
    'PEP', 'ADBE', 'CSCO', 'NFLX', 'AMD', 'TMUS', 'AMGN', 'INTC', 'QCOM', 'TXN',
    'HON', 'INTU', 'AMAT', 'BKNG', 'SBUX', 'ISRG', 'ADI', 'MDLZ', 'LRCX', 'REGN',
    'GILD', 'VRTX', 'ADP', 'MU', 'PDD', 'KDP', 'MAR', 'CSX', 'MELI', 'CTAS',
    'ASML', 'PANW', 'CDNS', 'ADSK', 'AEP', 'MNST', 'ORLY', 'SNPS', 'CRWD', 'FTNT',
    'DXCM', 'PAYX', 'MRVL', 'ODFL', 'PCAR', 'XEL', 'ROST', 'TEAM', 'LCID', 'SPLK',
    'DLTR', 'WBD', 'EXC', 'IDXX', 'FAST', 'EBAY', 'CEG', 'CHTR', 'VRSK', 'ANSS',
    'FANG', 'ZS', 'BKR', 'SIRI', 'TTD', 'CPRT', 'CTSH', 'KLAC', 'SWKS', 'GFS',
    'ON', 'BIDU', 'DDOG', 'ALGN', 'SGEN', 'VRSN', 'BIIB', 'MTCH', 'CDW', 'TTWO',
    'JD', 'GEN', 'SPLK', 'ZM', 'DOCU', 'VERV', 'PDD', 'ABNB', 'WDAY', 'LULU', 'ULTA'
]

def fetch_stock_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, auto_adjust=True)
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(i) for i in col if i]).strip('_') for col in df.columns.values]
    df.dropna(inplace=True)
    # Use 'Close' directly as auto_adjust handles splits/dividends
    if 'Close' in df.columns:
        df['Close_Single'] = df['Close']
    else:
        # fallback for single ticker with flattened columns
        df['Close_Single'] = df.filter(like='Close').iloc[:, 0]
    return df

def add_indicators(df):
    df['MA20'] = df['Close_Single'].rolling(window=20).mean()
    df['MA50'] = df['Close_Single'].rolling(window=50).mean()
    df['MA100'] = df['Close_Single'].rolling(window=100).mean()
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close_Single'].shift())
    low_close = np.abs(df['Low'] - df['Close_Single'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    return df

def generate_final_signals(df, params):
    short_ma_col = f"MA{params['short_ma']}"
    long_ma_col = f"MA{params['long_ma']}"
    
    # 1. Uptrend Regime Confirmation
    df['Uptrend_Regime'] = (df[short_ma_col] > df[long_ma_col]) & (df[long_ma_col] > df[long_ma_col].shift(1))

    # 2. Higher Low Structure Confirmation
    N = params['swing_low_window']
    df['is_swing_low'] = df['Low'] == df['Low'].rolling(2*N+1, center=True).min()
    swing_lows = df[df['is_swing_low']]['Low']
    
    # At each swing low day, check if it's higher than the previous one
    is_higher_low = swing_lows > swing_lows.shift(1)
    
    # Forward-fill this boolean state until the next swing low
    df['Higher_Low_Structure'] = is_higher_low.ffill().fillna(False)

    # 3. Buy on Dip Trigger
    buy_trigger = (df['Close_Single'] > df[short_ma_col]) & (df['Close_Single'].shift(1) <= df[short_ma_col].shift(1))
    
    # Final Buy Signal
    df['Buy_Signal'] = df['Uptrend_Regime'] & df['Higher_Low_Structure'] & buy_trigger
    
    df['Buy/Sell'] = np.where(df['Buy_Signal'], 'BUY', 'HOLD')
    return df

def backtest_final_strategy(df, params, initial_capital=10000):
    position = 0
    entry_price = 0
    cash = initial_capital
    holdings = 0
    trade_log = []
    max_high_since_entry = 0
    atr_multiplier = params['atr_trailing_stop_mult']

    for i in range(len(df)):
        date = df.index[i]
        price = df['Close_Single'].iloc[i]
        atr = df['ATR'].iloc[i] if pd.notna(df['ATR'].iloc[i]) else 0
        action = df['Buy/Sell'].iloc[i]

        if position == 1:
            max_high_since_entry = max(max_high_since_entry, price)

        if action == 'BUY' and position == 0:
            position = 1
            entry_price = price
            max_high_since_entry = price
            holdings = cash // price
            cash -= holdings * price
            trade_log.append((date, 'BUY', price, holdings, float('nan')))

        elif position == 1 and holdings > 0:
            trailing_stop = max_high_since_entry - atr * atr_multiplier
            if price < trailing_stop:
                gain = (price - entry_price) * holdings
                cash += holdings * price
                trade_log.append((date, 'SELL', price, holdings, gain))
                position = 0
                holdings = 0
                max_high_since_entry = 0

    if position == 1 and holdings > 0:
        final_price = df['Close_Single'].iloc[-1]
        gain = (final_price - entry_price) * holdings
        cash += holdings * final_price
        trade_log.append((df.index[-1], 'SELL', final_price, holdings, gain))
    
    trade_df = pd.DataFrame(trade_log, columns=["Date", "Action", "Price", "Shares", "Gain"])
    trade_df.to_csv("trade_log.csv", index=False)
    print("💾 Trade log saved to 'trade_log.csv'")

    final_value = cash
    profit = final_value - initial_capital
    pct = (profit / initial_capital) * 100
    trade_returns = trade_df[trade_df['Action'] == 'SELL']['Gain'] / (trade_df[trade_df['Action'] == 'SELL']['Price'] * trade_df[trade_df['Action'] == 'SELL']['Shares'])
    
    return {
        'final_value': final_value, 'profit': profit, 'pct': pct,
        'trades': len(trade_df[trade_df['Action']=='SELL']),
        'win_rate': (trade_df[trade_df['Action']=='SELL']['Gain'] > 0).sum() / len(trade_df[trade_df['Action']=='SELL']) * 100 if not trade_df.empty else 0,
        'avg_return': trade_returns.mean() * 100 if not trade_returns.empty else 0,
        'trade_log': trade_log,
    }

def plot_signals(df, symbol, trade_log, title):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Close_Single'], label='Close Price', alpha=0.6)
    plt.plot(df['MA20'], label='20-day MA', linestyle='--', alpha=0.7)
    plt.plot(df['MA50'], label='50-day MA', linestyle='--', alpha=0.7)
    plt.plot(df['MA100'], label='100-day MA', linestyle='--', alpha=0.5)

    if trade_log:
        trade_df = pd.DataFrame(trade_log, columns=["Date", "Action", "Price", "Shares", "Gain"])
        buy_trades = trade_df[trade_df['Action'] == 'BUY']
        sell_trades = trade_df[trade_df['Action'] == 'SELL']
        plt.scatter(buy_trades['Date'], buy_trades['Price'], marker='^', color='green', s=120, label='Executed Buy', zorder=5)
        plt.scatter(sell_trades['Date'], sell_trades['Price'], marker='v', color='red', s=120, label='Executed Sell', zorder=5)

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_col(df, base):
    for col in df.columns:
        if col.startswith(base):
            return col
    raise KeyError(f"No column starts with {base}")

def find_swing_lows(df, window=SWING_WINDOW):
    low_col = get_col(df, 'Low')
    df['is_swing_low'] = df[low_col] == df[low_col].rolling(2*window+1, center=True).min()
    swing_lows = df[df['is_swing_low']].copy()
    return swing_lows

def mark_higher_low_buys(df, swing_lows):
    low_col = get_col(swing_lows, 'Low')
    swing_lows = swing_lows.reset_index()
    if isinstance(swing_lows[low_col], pd.DataFrame):
        swing_lows[low_col] = swing_lows[low_col].iloc[:, 0]
    swing_lows[low_col] = swing_lows[low_col].astype(float)
    swing_lows['prev_low'] = swing_lows[low_col].shift(1)
    swing_lows['buy'] = swing_lows[low_col] >= swing_lows['prev_low']  # Allow equal or higher

    print("\n--- Swing Low Analysis ---")
    with pd.option_context('display.max_rows', None):
        print(swing_lows[['Date', low_col, 'prev_low', 'buy']])
    print("--------------------------\n")

    df['Buy'] = False
    buy_dates = swing_lows.loc[swing_lows['buy'], 'Date']
    df.loc[buy_dates, 'Buy'] = True

    print("\nBuy signals (higher low dips):")
    print(df[df['Buy']][[low_col]])
    return df

def find_swing_highs(df, window=SWING_WINDOW):
    high_col = get_col(df, 'High')
    df['is_swing_high'] = df[high_col] == df[high_col].rolling(2*window+1, center=True).max()
    swing_highs = df[df['is_swing_high']].copy()
    return swing_highs

def mark_swing_high_sells(df, swing_highs):
    df['Sell'] = False
    # For each buy, find the next swing high and mark as sell
    buy_indices = df.index[df['Buy']].tolist()
    swing_high_indices = swing_highs.index.tolist()
    sell_indices = []
    for buy_idx in buy_indices:
        # Find the first swing high after the buy
        next_swing_high = [idx for idx in swing_high_indices if idx > buy_idx]
        if next_swing_high:
            sell_idx = next_swing_high[0]
            sell_indices.append(sell_idx)
            df.loc[sell_idx, 'Sell'] = True
    return df

def plot_buys_and_sells(df, symbol):
    low_col = get_col(df, 'Low')
    high_col = get_col(df, 'High')
    plt.figure(figsize=(14, 7))
    plt.plot(df['Close_Single'], label='Close Price', alpha=0.7)
    plt.scatter(df[df['is_swing_low']].index, df[df['is_swing_low']][low_col], marker='o', color='gray', s=60, label='Swing Lows')
    plt.scatter(df[df['Buy']].index, df[df['Buy']][low_col], marker='^', color='green', s=120, label='Buy (Higher Low)')
    plt.scatter(df[df['is_swing_high']].index, df[df['is_swing_high']][high_col], marker='x', color='orange', s=80, label='Swing Highs')
    plt.scatter(df[df['Sell']].index, df[df['Sell']][high_col], marker='v', color='red', s=120, label='Sell (Swing High)')
    plt.title(f'{symbol} - Buy Dips (Higher than Previous Dip) & Sell at Swing Highs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def backtest_buy_sell_signals(df, initial_capital=10000):
    close_col = 'Close_Single'
    position = 0
    entry_price = 0
    cash = initial_capital
    shares = 0
    trade_log = []
    for i, row in df.iterrows():
        date = i
        price = row[close_col]
        if row.get('Buy', False) and position == 0:
            # Buy
            shares = cash // price
            entry_price = price
            cash -= shares * price
            position = 1
            trade_log.append((date, 'BUY', price, shares, float('nan')))
        elif row.get('Sell', False) and position == 1 and shares > 0:
            # Sell
            gain = (price - entry_price) * shares
            cash += shares * price
            trade_log.append((date, 'SELL', price, shares, gain))
            position = 0
            shares = 0
    # If still holding at the end, sell at last price
    if position == 1 and shares > 0:
        final_price = df[close_col].iloc[-1]
        gain = (final_price - entry_price) * shares
        cash += shares * final_price
        trade_log.append((df.index[-1], 'SELL', final_price, shares, gain))
    # Output trade log
    trade_df = pd.DataFrame(trade_log, columns=["Date", "Action", "Price", "Shares", "Gain"])
    trade_df.to_csv("trade_log.csv", index=False)
    print("\n💾 Trade log saved to 'trade_log.csv'")
    # Print stats
    final_value = cash
    profit = final_value - initial_capital
    pct = (profit / initial_capital) * 100
    print(f"Final value: ${final_value:.2f} | Profit: ${profit:.2f} | Return: {pct:.2f}%")
    print(trade_df)
    return trade_df, final_value, profit, pct

def scan_for_buys(symbols, start_date, end_date):
    buy_signals_today = []
    buy_signals_recent = []
    for symbol in symbols:
        try:
            df = fetch_stock_data(symbol, start_date, end_date)
            if df.empty:
                print(f"{symbol}: DataFrame is empty, skipping.")
                continue
            swing_lows = find_swing_lows(df)
            df = mark_higher_low_buys(df, swing_lows)
            print(f"{symbol}: last date {df.index[-1].date()}, last 5 Buy signals: {df['Buy'].tail().values}")
            # Check for buy signal today
            if df['Buy'].iloc[-1]:
                buy_signals_today.append((symbol, df.index[-1].date()))
                print(f"BUY signal for {symbol} on {df.index[-1].date()} (today)")
            # Check for buy signal in the last 5 days (excluding today)
            recent_window = min(5, len(df)-1)
            if recent_window > 0:
                recent_buys = df.iloc[-(recent_window+1):-1][df['Buy'].iloc[-(recent_window+1):-1]]
                for idx, row in recent_buys.iterrows():
                    buy_signals_recent.append((symbol, idx.date()))
                    print(f"BUY signal for {symbol} on {idx.date()} (recent 5 days)")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    print("\n=== BUY SIGNALS FOR TOMORROW (today) ===")
    print([f"{s} ({d})" for s, d in buy_signals_today])
    print("\n=== BUY SIGNALS IN LAST 5 DAYS (excluding today) ===")
    print([f"{s} ({d})" for s, d in buy_signals_recent])
    return buy_signals_today, buy_signals_recent

def scan_for_sells(symbols, start_date, end_date):
    sell_signals = []
    for symbol in symbols:
        try:
            df = fetch_stock_data(symbol, start_date, end_date)
            if df.empty:
                print(f"{symbol}: DataFrame is empty, skipping.")
                continue
            swing_lows = find_swing_lows(df)
            df = mark_higher_low_buys(df, swing_lows)
            swing_highs = find_swing_highs(df)
            df = mark_swing_high_sells(df, swing_highs)
            print(f"{symbol}: last date {df.index[-1].date()}, last 5 Sell signals: {df['Sell'].tail().values}")
            if df['Sell'].iloc[-1]:
                sell_signals.append(symbol)
                print(f"SELL signal for {symbol} on {df.index[-1].date()}")
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    print("\n=== SELL SIGNALS FOR TOMORROW ===")
    print(sell_signals)
    return sell_signals

def read_tickers_from_file(filename):
    with open(filename, 'r') as f:
        return [line.strip().upper() for line in f if line.strip()]

def plot_stock_by_name():
    import datetime
    symbol = input("Enter stock symbol to plot (e.g. TSLA): ").strip().upper()
    start_date = input("Enter start date (YYYY-MM-DD, default 2025-01-01): ").strip() or "2025-01-01"
    end_date = input("Enter end date (YYYY-MM-DD, default today): ").strip() or datetime.datetime.today().strftime('%Y-%m-%d')
    df = fetch_stock_data(symbol, start_date, end_date)
    if df.empty:
        print(f"No data for {symbol}")
        return
    swing_lows = find_swing_lows(df)
    df = mark_higher_low_buys(df, swing_lows)
    swing_highs = find_swing_highs(df)
    df = mark_swing_high_sells(df, swing_highs)
    plot_buys_and_sells(df, symbol)

def main():
    start_date = "2025-01-01"
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    my_tickers = read_tickers_from_file('my_stocks.txt')
    buy_today, buy_recent = scan_for_buys(NASDAQ_100, start_date, end_date)
    sell_list = scan_for_sells(my_tickers, start_date, end_date)
    print("\n=== SUMMARY ===")
    print("Buy tomorrow (today):", buy_today)
    print("Buy signal in last 5 days (excluding today):", buy_recent)
    print("Sell tomorrow:", sell_list)
    Prompt user to plot a stock
    # plot_stock_by_name()

if __name__ == "__main__":
    main()
