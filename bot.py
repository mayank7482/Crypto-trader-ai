# ==========================================================
# PRO AI CRYPTO TERMINAL BOT (CODESPACES VERSION)
# Clean dashboard • colored UI • ML learning • persistent
# ==========================================================

import requests, time, math, os
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live

console = Console()

# ================= SETTINGS =================
START_BAL = 150
REFRESH = 6
HISTORY_FILE = "trade_history.csv"

SYMBOLS = [
    "BTC","ETH","SOL","XRP","ADA","DOGE","DOT",
    "MATIC","LTC","LINK","AVAX","ATOM","BCH"
]
# ============================================

balance = START_BAL
portfolio = {}
entry = {}
price_history = {}
wins = trades = 0
risk = 0.15
logs = []

model = LogisticRegression()

# ============================================
def ist():
    return (datetime.utcnow()+timedelta(hours=5,minutes=30)).strftime("%H:%M:%S IST")

# ============================================
# DATA SOURCES
# ============================================
def prices():
    coins=[]
    for s in SYMBOLS:
        try:
            r=requests.get(f"https://api.coinbase.com/v2/prices/{s}-USD/spot")
            p=float(r.json()["data"]["amount"])
            coins.append((s,p))
        except:
            pass
    return coins


def news():
    try:
        r=requests.get(
            "https://www.reddit.com/r/cryptocurrency/hot.json?limit=5",
            headers={"User-agent":"bot"}
        ).json()
        return [x["data"]["title"] for x in r["data"]["children"]]
    except:
        return []


# ============================================
# HISTORY (self learning)
# ============================================
if os.path.exists(HISTORY_FILE):
    history=pd.read_csv(HISTORY_FILE)
else:
    history=pd.DataFrame(columns=["mom","trend","sent","result"])

def save():
    history.to_csv(HISTORY_FILE,index=False)


# ============================================
# FEATURES
# ============================================
def sentiment_score(newslist):
    s=0
    for n in newslist:
        n=n.lower()
        if any(w in n for w in ["bull","rise","gain","up"]): s+=1
        if any(w in n for w in ["crash","drop","hack","down"]): s-=1
    return s


def features(sym,sent):
    h=price_history.get(sym,[])
    if len(h)<6: return None
    return [h[-1]-h[-2], h[-1]-h[-5], sent]


def train():
    if len(history)<30: return False
    model.fit(history[["mom","trend","sent"]],history["result"])
    return True


# ============================================
# AI TRADER
# ============================================
def ai_trade(coins,newslist):
    global balance,wins,trades,risk,history

    sent=sentiment_score(newslist)
    trained=train()

    best=None
    best_prob=0

    for sym,price in coins:
        f=features(sym,sent)
        if not f: continue

        prob=model.predict_proba([f])[0][1] if trained else 0.5+math.tanh(sum(f)/20)/2

        if prob>best_prob:
            best_prob=prob
            best=(sym,price,f)

    if not best: return

    sym,price,f=best

    # BUY
    if sym not in portfolio and best_prob>0.65 and balance>10:
        qty=(balance*risk)/price
        balance-=qty*price
        portfolio[sym]=qty
        entry[sym]=price
        logs.append(f"[{ist()}] BUY {sym} prob {best_prob:.2f}")
        return

    # SELL
    if sym in portfolio:
        qty=portfolio[sym]
        pnl=(price-entry[sym])*qty

        if best_prob<0.45 or pnl>1:
            balance+=qty*price
            win=int(pnl>0)

            history.loc[len(history)]=f+[win]
            save()

            trades+=1
            wins+=win

            logs.append(f"[{ist()}] SELL {sym} pnl {pnl:.2f}")

            del portfolio[sym]
            del entry[sym]

            risk=max(0.05,min(0.25,wins/trades if trades else 0.5))


# ============================================
# UI BUILDERS
# ============================================
def build_table(coins):
    t=Table(title="📈 Market Scanner")
    t.add_column("Coin")
    t.add_column("Price",justify="right")

    for s,p in coins:
        color="green" if len(price_history[s])>1 and p>price_history[s][-2] else "red"
        t.add_row(s,f"[{color}]${p:.2f}[/]")

    return t


def build_portfolio(coins):
    prices=dict(coins)
    t=Table(title="💼 Portfolio")

    t.add_column("Coin")
    t.add_column("Value",justify="right")
    t.add_column("PnL",justify="right")

    for s,q in portfolio.items():
        val=q*prices[s]
        pnl=(prices[s]-entry[s])*q
        color="green" if pnl>=0 else "red"
        t.add_row(s,f"${val:.2f}",f"[{color}]{pnl:.2f}[/]")

    return t


def build_news(newslist):
    text="\n".join("• "+n for n in newslist[:5])
    return Panel(text,title="📰 News Highlights")


def build_logs():
    return Panel("\n".join(logs[-6:]),title="📜 Logs")


# ============================================
# MAIN LOOP
# ============================================
with Live(refresh_per_second=4) as live:

    while True:
        coins=prices()
        newslist=news()

        for s,p in coins:
            price_history.setdefault(s,[]).append(p)

        ai_trade(coins,newslist)

        header=Panel(
            f"💰 Balance ${balance:.2f} | Trades {trades} | Win {(wins/trades*100 if trades else 0):.1f}% | {ist()}",
            style="bold cyan"
        )

        layout=Table.grid(expand=True)
        layout.add_row(header)
        layout.add_row(build_table(coins))
        layout.add_row(build_portfolio(coins))
        layout.add_row(build_news(newslist))
        layout.add_row(build_logs())

        live.update(layout)

        time.sleep(REFRESH)
