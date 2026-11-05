#!/usr/bin/env python3
"""S&P 500 每日市场宽度监测。

拉取成分股日线 → 计算宽度指标 → 生成可交互 HTML（ECharts）。

用法:
    python breadth.py                    # 拉取约 3 年数据，输出 sp500_breadth.html
    python breadth.py --out report.html --years 4
    python breadth.py --selftest         # 用随机数据检查计算与页面（页面会标注“测试数据”）

数据源: 成分股名单来自 github.com/datasets/s-and-p-500-companies；
价格与成交量来自 Yahoo Finance (yfinance)，复权收盘价。
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CONSTITUENTS_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
INDEX_TICKER = "^GSPC"
DISPLAY_DAYS = 504          # 页面展示约 2 年
HEATMAP_DAYS = 60

SECTOR_CN = {
    "Information Technology": "信息技术", "Health Care": "医疗保健", "Financials": "金融",
    "Consumer Discretionary": "可选消费", "Communication Services": "通信服务",
    "Industrials": "工业", "Consumer Staples": "必需消费", "Energy": "能源",
    "Utilities": "公用事业", "Real Estate": "房地产", "Materials": "原材料",
}


# ---------------------------------------------------------------- 数据获取
def get_constituents() -> pd.DataFrame:
    req = urllib.request.Request(CONSTITUENTS_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        df = pd.read_csv(io.BytesIO(r.read()))
    df = df.rename(columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector"})
    df["yf"] = df["ticker"].str.replace(".", "-", regex=False)   # BRK.B → BRK-B
    return df[["ticker", "yf", "name", "sector"]]


def get_prices(tickers: list[str], years: int, batch: int = 100):
    import yfinance as yf
    closes, vols = [], []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        for attempt in range(4):
            try:
                d = yf.download(chunk, period=f"{years}y", interval="1d", auto_adjust=True,
                                progress=False, threads=True, group_by="column")
                break
            except Exception as e:  # 网络抖动重试
                print(f"  batch {i // batch} retry {attempt + 1}: {e}", file=sys.stderr)
                time.sleep(2 ** (attempt + 1))
        else:
            raise RuntimeError("Yahoo Finance 下载失败")
        closes.append(d["Close"])
        vols.append(d["Volume"])
        print(f"  已下载 {min(i + batch, len(tickers))}/{len(tickers)}", file=sys.stderr)
    close = pd.concat(closes, axis=1)
    vol = pd.concat(vols, axis=1)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    vol.index = close.index
    return close, vol


def get_index_ohlc(years: int) -> pd.DataFrame:
    """标普500指数日K：Open/High/Low/Close。"""
    import yfinance as yf
    for attempt in range(4):
        try:
            d = yf.download(INDEX_TICKER, period=f"{years}y", interval="1d", auto_adjust=False, progress=False)
            if len(d):
                break
        except Exception as e:
            print(f"  index retry {attempt + 1}: {e}", file=sys.stderr)
        time.sleep(2 ** (attempt + 1))
    else:
        raise RuntimeError("标普500指数下载失败")
    if isinstance(d.columns, pd.MultiIndex):
        d = d.xs(INDEX_TICKER, axis=1, level=-1)
    d = d[["Open", "High", "Low", "Close"]]
    d.index = pd.to_datetime(d.index).tz_localize(None)
    return d


# ---------------------------------------------------------------- 指标计算
def compute(close: pd.DataFrame, vol: pd.DataFrame, sectors: pd.Series) -> dict:
    """close/vol: 日期 × 股票；sectors: 股票 → 板块。"""
    close = close.dropna(how="all")
    vol = vol.reindex(close.index)
    ret = close.pct_change(fill_method=None)
    has = ret.notna()

    adv = (ret > 0).sum(axis=1)
    dec = (ret < 0).sum(axis=1)
    unch = (has & (ret == 0)).sum(axis=1)
    issues = adv + dec
    net = adv - dec
    ad_line = net.cumsum()

    # McClellan（比例调整版：消除成分数量变化影响）
    rana = (net / issues.replace(0, np.nan) * 1000).fillna(0)
    mco = rana.ewm(span=19, adjust=False).mean() - rana.ewm(span=39, adjust=False).mean()
    msi = mco.cumsum()

    # 上涨成交量占比
    upvol = vol.where(ret > 0).sum(axis=1)
    totvol = vol.where(has).sum(axis=1)
    upvol_pct = upvol / totvol.replace(0, np.nan) * 100

    # 站上均线比例
    def pct_above(n):
        ma = close.rolling(n, min_periods=n).mean()
        valid = ma.notna() & close.notna()
        return ((close > ma) & valid).sum(axis=1) / valid.sum(axis=1).replace(0, np.nan) * 100, ma

    p20, _ = pct_above(20)
    p50, ma50 = pct_above(50)
    p200, _ = pct_above(200)

    # 52 周新高 / 新低（收盘口径）
    hi = close.rolling(252, min_periods=200).max()
    lo = close.rolling(252, min_periods=200).min()
    nh = ((close >= hi) & hi.notna()).sum(axis=1)
    nl = ((close <= lo) & lo.notna()).sum(axis=1)

    # Zweig 宽度推力：10 日 EMA(上涨家数/总家数) 10 个交易日内从 <40% 升至 >61.5%
    ratio = (adv / issues.replace(0, np.nan)).ewm(span=10, adjust=False).mean()
    thrust = (ratio > 0.615) & (ratio.rolling(11).min() < 0.40)
    thrust = thrust & ~thrust.shift(1, fill_value=False)

    # 板块：站上 50 日线比例
    above50 = (close > ma50) & ma50.notna()
    valid50 = ma50.notna() & close.notna()
    sec = sectors.reindex(close.columns)
    sec_p50 = {}
    sec_tbl = []
    for s in sorted(sec.dropna().unique()):
        cols = sec.index[sec == s]
        series = above50[cols].sum(axis=1) / valid50[cols].sum(axis=1).replace(0, np.nan) * 100
        sec_p50[s] = series
        def last_pct(n):
            ma = close[cols].rolling(n, min_periods=n).mean()
            v = ma.notna() & close[cols].notna()
            r = ((close[cols] > ma) & v).sum(axis=1) / v.sum(axis=1).replace(0, np.nan) * 100
            return r
        r20, r200 = last_pct(20), last_pct(200)
        r1m = (close[cols].iloc[-1] / close[cols].iloc[-22] - 1).median() * 100 if len(close) > 22 else np.nan
        sec_tbl.append({
            "sector": s, "cn": SECTOR_CN.get(s, s), "n": int(len(cols)),
            "p20": r20.iloc[-1], "p50": series.iloc[-1], "p200": r200.iloc[-1],
            "p50_chg5": series.iloc[-1] - series.iloc[-6] if len(series) > 6 else np.nan,
            "ret1m": r1m,
        })

    df = pd.DataFrame({
        "adv": adv, "dec": dec, "unch": unch, "net": net, "ad": ad_line,
        "mco": mco, "msi": msi, "upvol": upvol_pct,
        "p20": p20, "p50": p50, "p200": p200, "nh": nh, "nl": nl,
        "thrust": thrust.astype(int),
    })
    sec_df = pd.DataFrame(sec_p50)
    return {"daily": df, "sector_p50": sec_df, "sector_table": sec_tbl}


# ---------------------------------------------------------------- 渲染
def _r(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x):
        return None
    return round(float(x), nd)


def build_payload(res: dict, index_ohlc: pd.DataFrame, meta: dict) -> dict:
    d = res["daily"].tail(DISPLAY_DAYS)
    idx = index_ohlc.reindex(d.index).ffill()
    hm = res["sector_p50"].tail(HEATMAP_DAYS)
    tbl = sorted(res["sector_table"], key=lambda r: -(r["p50"] if pd.notna(r["p50"]) else -1))
    return {
        "meta": meta,
        "dates": [t.strftime("%Y-%m-%d") for t in d.index],
        # ECharts K线顺序：[开, 收, 低, 高]
        "spx": [[_r(r.Open), _r(r.Close), _r(r.Low), _r(r.High)] for r in idx.itertuples()],
        **{k: [_r(v) for v in d[k]] for k in ["p20", "p50", "p200", "mco", "msi", "upvol"]},
        **{k: [int(v) for v in d[k]] for k in ["adv", "dec", "unch", "net", "ad", "nh", "nl", "thrust"]},
        "heat": {
            "dates": [t.strftime("%m-%d") for t in hm.index],
            "sectors": [SECTOR_CN.get(s, s) for s in hm.columns],
            "values": [[i, j, _r(hm.iloc[i, j], 1)] for i in range(len(hm)) for j in range(hm.shape[1])],
        },
        "sectors": [{k: (_r(v, 1) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()} for r in tbl],
    }


def render(payload: dict, out: Path):
    tpl = TEMPLATE
    html = tpl.replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    out.write_text(html, encoding="utf-8")
    print(f"已生成 {out.resolve()}  ({out.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------- 自测数据
def synthetic(n_stocks=500, n_days=760, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    secs = list(SECTOR_CN)
    sector = pd.Series([secs[i % len(secs)] for i in range(n_stocks)], index=[f"T{i:03d}" for i in range(n_stocks)])
    mkt = rng.normal(0.0004, 0.010, n_days)
    sec_f = {s: rng.normal(0, 0.006, n_days) for s in secs}
    beta = rng.uniform(0.6, 1.5, n_stocks)
    rets = np.column_stack([beta[i] * mkt + sec_f[sector.iloc[i]] + rng.normal(0, 0.014, n_days)
                            for i in range(n_stocks)])
    close = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=sector.index)
    vol = pd.DataFrame(rng.lognormal(14, 0.4, close.shape), index=dates, columns=sector.index)
    c = 4000 * np.exp(np.cumsum(mkt))
    o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.002, n_days))
    rng_hl = np.abs(rng.normal(0, 0.006, n_days)) * c
    spx = pd.DataFrame({"Open": o, "Close": c, "High": np.maximum(o, c) + rng_hl,
                        "Low": np.minimum(o, c) - rng_hl}, index=dates)
    return close, vol, sector, spx


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="sp500_breadth.html")
    ap.add_argument("--years", type=int, default=3, help="下载年数（需 ≥2，留出 200 日线与 52 周窗口）")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--csv", help="另存每日宽度指标 CSV（可选）")
    ap.add_argument("--min-stocks", type=int, default=400, help="成功下载的股票少于此数则报错退出")
    a = ap.parse_args()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if a.selftest:
        close, vol, sector, spx = synthetic()
        meta = {"test": True, "generated": now, "n": int(close.shape[1]), "source": "随机游走测试数据"}
    else:
        cons = get_constituents()
        print(f"成分股 {len(cons)} 只，开始下载 {a.years} 年日线…", file=sys.stderr)
        close, vol = get_prices(cons["yf"].tolist(), a.years)
        spx = get_index_ohlc(a.years)
        sector = cons.set_index("yf")["sector"]
        got = close.columns[close.notna().sum() > 0]
        if len(got) < a.min_stocks:
            sys.exit(f"只下载到 {len(got)} 只股票（少于 {a.min_stocks}），可能被 Yahoo 限流，放弃本次更新")
        close, vol = close[got], vol[got]
        meta = {"test": False, "generated": now, "n": int(len(got)),
                "source": "Yahoo Finance 复权日线 · 成分股名单 datasets/s-and-p-500-companies"}

    res = compute(close, vol, sector)
    payload = build_payload(res, spx, meta)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    render(payload, out)
    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        res["daily"].round(2).to_csv(a.csv, index_label="date")
    last = res["daily"].iloc[-1]
    print(f"{res['daily'].index[-1].date()}  涨 {int(last.adv)} / 跌 {int(last.dec)}  "
          f">20DMA {last.p20:.1f}%  >50DMA {last.p50:.1f}%  >200DMA {last.p200:.1f}%  "
          f"McClellan {last.mco:.1f}  新高 {int(last.nh)} / 新低 {int(last.nl)}")


TEMPLATE = r'''<title>标普500市场宽度</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;700;800&family=IBM+Plex+Mono:wght@400;500&family=Noto+Sans+SC:wght@400;500;700&display=swap">
<style>
/* Layout: 一张盘后行情单 — 顶部读数条 → 主图（指数 vs 均线宽度）→ 四个副图双列 → 板块热力 + 表格 */
:root{
  --bg:#eef1f4; --panel:#ffffff; --ink:#15202b; --muted:#5d6b78; --line:#d5dbe1; --grid:#e6eaee;
  --accent:#1f5f8b; --c20:#d08a1e; --c50:#1f8a8a; --c200:#5b4bb7;
  --red:#d03b32; --green:#1e8a4c; --warn:#b7791f; --chip:#e4e9ee;
  --f-display:"Archivo","Noto Sans SC",system-ui,sans-serif;
  --f-body:"Noto Sans SC","PingFang SC","Microsoft YaHei",system-ui,sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,"SFMono-Regular",Menlo,monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --bg:#0d141b; --panel:#141d26; --ink:#e3e9ef; --muted:#8d9aa7; --line:#25313d; --grid:#1d2832;
  --accent:#6fb3e0; --c20:#e5a54a; --c50:#45c2bd; --c200:#9d8ff0;
  --red:#ff6b5e; --green:#3fc27a; --warn:#e2a64a; --chip:#1f2a35; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#0d141b; --panel:#141d26; --ink:#e3e9ef; --muted:#8d9aa7; --line:#25313d; --grid:#1d2832;
  --accent:#6fb3e0; --c20:#e5a54a; --c50:#45c2bd; --c200:#9d8ff0;
  --red:#ff6b5e; --green:#3fc27a; --warn:#e2a64a; --chip:#1f2a35; color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:var(--f-body);font-size:14px;line-height:1.5}
.wrap{max-width:1320px;margin:0 auto;padding-block:20px 40px;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));
  display:flex;flex-direction:column;gap:16px}
header{display:flex;flex-wrap:wrap;align-items:flex-end;justify-content:space-between;gap:12px 24px}
h1{font-family:var(--f-display);font-weight:800;font-size:clamp(22px,3vw,30px);letter-spacing:-.01em;margin:0;text-wrap:balance}
h1 small{font-family:var(--f-mono);font-weight:500;font-size:13px;color:var(--muted);margin-left:10px;letter-spacing:.04em}
.sub{color:var(--muted);font-size:12.5px;margin-top:4px}
.test{display:inline-block;background:var(--warn);color:var(--panel);font-weight:700;padding:1px 8px;border-radius:3px;margin-right:6px}
.controls{display:flex;flex-wrap:wrap;gap:8px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden;background:var(--panel)}
.seg button{font:500 12.5px var(--f-mono);color:var(--muted);background:none;border:0;padding:6px 11px;cursor:pointer}
.seg button+button{border-left:1px solid var(--line)}
.seg button[aria-pressed="true"]{background:var(--ink);color:var(--panel)}
.seg button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.kpis{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.kpi{background:var(--panel);padding:12px 14px;display:flex;flex-direction:column;gap:4px;min-width:0}
.kpi .lab{font-size:12px;color:var(--muted);letter-spacing:.02em}
.kpi .val{font:500 24px/1.1 var(--f-mono);font-variant-numeric:tabular-nums}
.kpi .row{display:flex;align-items:center;justify-content:space-between;gap:6px;font:12px var(--f-mono);color:var(--muted)}
.chipc{font:500 11.5px var(--f-body);padding:1px 7px;border-radius:999px;white-space:nowrap}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 14px 6px;min-width:0}
.panel h2{font:700 15px var(--f-display);margin:0;letter-spacing:.01em}
.panel .note{color:var(--muted);font-size:12px;margin:2px 0 6px}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.chart{width:100%;height:300px}
.chart.tall{height:440px}
.tbl{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:560px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--grid);white-space:nowrap}
th{font-weight:500;color:var(--muted);font-size:12px}
th:first-child,td:first-child{text-align:left}
td.num{font-family:var(--f-mono);font-variant-numeric:tabular-nums}
.bar{display:inline-block;height:6px;border-radius:3px;background:var(--c50);vertical-align:middle;margin-left:8px;opacity:.75}
footer{color:var(--muted);font-size:12px;line-height:1.7}
footer b{color:var(--ink);font-weight:500}
@media (max-width:1000px){.kpis{grid-template-columns:repeat(3,minmax(0,1fr))}.grid2{grid-template-columns:minmax(0,1fr)}}
@media (max-width:520px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}.chart.tall{height:380px}.kpi .val{font-size:20px}}
</style>

<div class="wrap">
  <header>
    <div>
      <h1>标普500市场宽度<small id="asof"></small></h1>
      <div class="sub" id="src"></div>
    </div>
    <div class="controls">
      <div class="seg" id="range" role="group" aria-label="时间范围">
        <button data-d="63">3月</button><button data-d="126">6月</button><button data-d="252" aria-pressed="true">1年</button><button data-d="0">全部</button>
      </div>
      <div class="seg" id="conv" role="group" aria-label="涨跌配色">
        <button data-c="cn" aria-pressed="true">红涨绿跌</button><button data-c="us">绿涨红跌</button>
      </div>
    </div>
  </header>

  <section class="kpis" id="kpis" aria-label="最新读数"></section>

  <section class="panel">
    <h2>指数与均线宽度</h2>
    <p class="note">上：标普500指数日K线。下：站上 20 / 50 / 200 日均线的成分股比例，80% 以上为过热区、20% 以下为超卖区。拖动底部滑块缩放。</p>
    <div class="chart tall" id="c-main"></div>
  </section>

  <div class="grid2">
    <section class="panel">
      <h2>腾落线与 McClellan 振荡器</h2>
      <p class="note">腾落线 = 每日（上涨家数 − 下跌家数）累计；柱为比例调整 McClellan 振荡器（19/39 日 EMA 差）。</p>
      <div class="chart" id="c-ad"></div>
    </section>
    <section class="panel">
      <h2>52 周新高 / 新低</h2>
      <p class="note">以收盘价计。柱向上为创新高家数，向下为创新低家数；线为净新高 10 日均值。</p>
      <div class="chart" id="c-hl"></div>
    </section>
    <section class="panel">
      <h2>每日涨跌家数</h2>
      <p class="note">堆叠柱为上涨 / 下跌家数；线为上涨成交量占总成交量比例（右轴）。▲ 标记 Zweig 宽度推力信号。</p>
      <div class="chart" id="c-adv"></div>
    </section>
    <section class="panel">
      <h2>McClellan 累积指数</h2>
      <p class="note">振荡器的累计值，衡量中期宽度趋势；在 0 上方且上行为偏多。</p>
      <div class="chart" id="c-msi"></div>
    </section>
  </div>

  <section class="panel">
    <h2>板块宽度热力图</h2>
    <p class="note">近 60 个交易日，各 GICS 板块站上 50 日均线的成分股比例。</p>
    <div class="chart" id="c-heat" style="height:400px"></div>
  </section>

  <section class="panel">
    <h2>板块快照</h2>
    <p class="note">按站上 50 日线比例排序。5 日变化为百分点；1 月中位涨幅为板块内个股近 21 个交易日涨幅的中位数。</p>
    <div class="tbl"><table id="tbl"></table></div>
  </section>

  <footer id="foot"></footer>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.5.0/echarts.min.js"></script>
<script>
const D = /*__DATA__*/null;
const N = D.dates.length;
let range = 252, conv = 'cn';
try { conv = localStorage.getItem('breadth-conv') || 'cn'; } catch (e) {}

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const T = () => ({
  ink: css('--ink'), muted: css('--muted'), line: css('--line'), grid: css('--grid'), panel: css('--panel'),
  accent: css('--accent'), c20: css('--c20'), c50: css('--c50'), c200: css('--c200'), warn: css('--warn'),
  up: conv === 'cn' ? css('--red') : css('--green'), down: conv === 'cn' ? css('--green') : css('--red'),
  mono: '"IBM Plex Mono", ui-monospace, monospace',
});
const last = a => { for (let i = a.length - 1; i >= 0; i--) if (a[i] != null) return a[i]; return null; };
const ago = (a, k) => a[a.length - 1 - k];
const fmt = (v, d = 1) => v == null ? '—' : v.toFixed(d);
const sign = (v, d = 1) => v == null ? '—' : (v > 0 ? '+' : '') + v.toFixed(d);
const ma = (a, n) => a.map((_, i) => i < n - 1 ? null : a.slice(i - n + 1, i + 1).reduce((s, x) => s + x, 0) / n);

/* ------------ header / KPI ------------ */
document.getElementById('asof').textContent = '截至 ' + D.dates[N - 1];
document.getElementById('src').innerHTML = (D.meta.test ? '<span class="test">测试数据</span>' : '') +
  `${D.meta.n} 只成分股 · ${D.meta.source} · 生成于 ${D.meta.generated}`;

function state(v, lo, hi, rev) {
  const t = T();
  if (v == null) return ['—', t.muted];
  if (v >= hi) return rev ? ['过热', t.warn] : ['强', t.up];
  if (v <= lo) return rev ? ['超卖', t.warn] : ['弱', t.down];
  return ['中性', t.muted];
}
function kpis() {
  const t = T();
  const net = D.nh[N - 1] - D.nl[N - 1];
  const items = [
    { lab: '站上 20 日线', v: fmt(last(D.p20)) + '%', d: sign(last(D.p20) - ago(D.p20, 5)) + ' pp / 5日', s: state(last(D.p20), 20, 80, true) },
    { lab: '站上 50 日线', v: fmt(last(D.p50)) + '%', d: sign(last(D.p50) - ago(D.p50, 5)) + ' pp / 5日', s: state(last(D.p50), 30, 70) },
    { lab: '站上 200 日线', v: fmt(last(D.p200)) + '%', d: sign(last(D.p200) - ago(D.p200, 5)) + ' pp / 5日', s: state(last(D.p200), 40, 60) },
    { lab: '今日涨 / 跌', v: `${D.adv[N - 1]} / ${D.dec[N - 1]}`, d: '上涨量 ' + fmt(last(D.upvol), 0) + '%', s: state(D.adv[N - 1] / (D.adv[N - 1] + D.dec[N - 1]) * 100, 35, 65) },
    { lab: 'McClellan 振荡器', v: fmt(last(D.mco)), d: sign(last(D.mco) - ago(D.mco, 1)) + ' / 1日', s: last(D.mco) > 0 ? ['偏多', t.up] : ['偏空', t.down] },
    { lab: '52周新高 / 新低', v: `${D.nh[N - 1]} / ${D.nl[N - 1]}`, d: '净 ' + sign(net, 0), s: net > 0 ? ['偏多', t.up] : net < 0 ? ['偏空', t.down] : ['持平', t.muted] },
  ];
  document.getElementById('kpis').innerHTML = items.map(k => `
    <div class="kpi"><span class="lab">${k.lab}</span><span class="val">${k.v}</span>
    <span class="row"><span>${k.d}</span><span class="chipc" style="color:${k.s[1]};background:color-mix(in srgb, ${k.s[1]} 14%, transparent)">${k.s[0]}</span></span></div>`).join('');
}

/* ------------ charts ------------ */
const charts = {};
function base(t, extra = {}) {
  return Object.assign({
    animation: false, backgroundColor: 'transparent',
    textStyle: { color: t.muted, fontFamily: t.mono, fontSize: 11 },
    tooltip: { trigger: 'axis', axisPointer: { type: 'line', lineStyle: { color: t.muted, type: 'dashed' } },
      backgroundColor: t.panel, borderColor: t.line, textStyle: { color: t.ink, fontFamily: t.mono, fontSize: 12 },
      valueFormatter: v => v == null ? '—' : (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : +(+v).toFixed(2)) },
    legend: { top: 0, right: 0, textStyle: { color: t.muted, fontSize: 11 }, itemWidth: 14, itemHeight: 8 },
  }, extra);
}
const xAx = (t, gi = 0, show = true) => ({ type: 'category', data: D.dates, gridIndex: gi, boundaryGap: false,
  axisLine: { lineStyle: { color: t.line } }, axisTick: { show: false }, axisLabel: { show, color: t.muted, hideOverlap: true },
  axisPointer: { label: { show: false } } });
const yAx = (t, o = {}) => Object.assign({ type: 'value', scale: true, splitNumber: 4,
  axisLabel: { color: t.muted }, splitLine: { lineStyle: { color: t.grid } }, axisLine: { show: false } }, o);
const zoom = (idx) => [{ type: 'inside', xAxisIndex: idx, startValue: Math.max(0, N - (range || N)), endValue: N - 1 }];
const line = (name, data, color, o = {}) => Object.assign({ name, type: 'line', data, showSymbol: false, lineStyle: { width: 1.6, color }, itemStyle: { color } }, o);

function drawAll() {
  const t = T();
  const mk = (id) => { if (!charts[id]) charts[id] = echarts.init(document.getElementById(id)); return charts[id]; };

  mk('c-main').setOption(base(t, {
    grid: [{ left: 8, right: 12, top: 28, height: '38%', containLabel: true }, { left: 8, right: 12, top: '52%', bottom: 58, containLabel: true }],
    xAxis: [xAx(t, 0, false), xAx(t, 1)],
    yAxis: [yAx(t, { gridIndex: 0, axisLabel: { color: t.muted, formatter: v => Math.round(v).toLocaleString() } }),
            yAx(t, { gridIndex: 1, min: 0, max: 100, scale: false, interval: 20, axisLabel: { color: t.muted, formatter: '{value}%' } })],
    axisPointer: { link: [{ xAxisIndex: 'all' }] },
    tooltip: Object.assign(base(t).tooltip, { formatter: ps => {
      const i = ps[0].dataIndex, k = D.spx[i], pc = i > 0 ? D.spx[i - 1][1] : null;
      const chg = pc ? (k[1] / pc - 1) * 100 : null, n = v => v == null ? '—' : v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
      const cc = chg == null ? t.ink : chg >= 0 ? t.up : t.down;
      const row = (c, l, v) => `<div style="display:flex;justify-content:space-between;gap:16px"><span>${c ? `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${c};margin-right:6px"></span>` : ''}${l}</span><b>${v}</b></div>`;
      return `<div style="margin-bottom:4px">${D.dates[i]}</div>` +
        row(null, '开', n(k[0])) + row(null, '高', n(k[3])) + row(null, '低', n(k[2])) +
        `<div style="display:flex;justify-content:space-between;gap:16px"><span>收</span><b style="color:${cc}">${n(k[1])}　${sign(chg, 2)}%</b></div>` +
        `<div style="border-top:1px solid ${t.line};margin:4px 0"></div>` +
        row(t.c20, '>20日线', fmt(D.p20[i]) + '%') + row(t.c50, '>50日线', fmt(D.p50[i]) + '%') + row(t.c200, '>200日线', fmt(D.p200[i]) + '%');
    } }),
    dataZoom: [...zoom([0, 1]), { type: 'slider', xAxisIndex: [0, 1], bottom: 8, height: 22, borderColor: t.line,
      textStyle: { color: t.muted }, fillerColor: 'rgba(127,127,127,.12)', dataBackground: { lineStyle: { color: t.muted }, areaStyle: { color: t.grid } },
      startValue: Math.max(0, N - (range || N)), endValue: N - 1 }],
    series: [
      { name: '标普500', type: 'candlestick', data: D.spx, barMaxWidth: 10,
        itemStyle: { color: t.up, color0: t.down, borderColor: t.up, borderColor0: t.down, borderWidth: 1 } },
      line('>20日线', D.p20, t.c20, { xAxisIndex: 1, yAxisIndex: 1, lineStyle: { width: 1, color: t.c20, opacity: .8 },
        markArea: { silent: true, itemStyle: { color: t.warn, opacity: .07 }, data: [[{ yAxis: 80 }, { yAxis: 100 }], [{ yAxis: 0 }, { yAxis: 20 }]] } }),
      line('>50日线', D.p50, t.c50, { xAxisIndex: 1, yAxisIndex: 1, lineStyle: { width: 1.8, color: t.c50 } }),
      line('>200日线', D.p200, t.c200, { xAxisIndex: 1, yAxisIndex: 1, lineStyle: { width: 2.2, color: t.c200 } }),
    ],
  }), true);

  mk('c-ad').setOption(base(t, {
    grid: { left: 8, right: 8, top: 28, bottom: 28, containLabel: true },
    xAxis: xAx(t), yAxis: [yAx(t), yAx(t, { position: 'right', splitLine: { show: false } })],
    dataZoom: zoom([0]),
    series: [
      { name: 'McClellan', type: 'bar', yAxisIndex: 1, data: D.mco.map(v => ({ value: v, itemStyle: { color: v >= 0 ? t.up : t.down, opacity: .55 } })), barCategoryGap: '10%' },
      line('腾落线', D.ad, t.ink, { lineStyle: { width: 1.6, color: t.ink } }),
    ],
  }), true);

  const net10 = ma(D.nh.map((v, i) => v - D.nl[i]), 10);
  mk('c-hl').setOption(base(t, {
    grid: { left: 8, right: 8, top: 28, bottom: 28, containLabel: true },
    xAxis: xAx(t), yAxis: yAx(t, { scale: false }), dataZoom: zoom([0]),
    series: [
      { name: '新高', type: 'bar', stack: 'hl', data: D.nh, itemStyle: { color: t.up, opacity: .7 }, barCategoryGap: '15%' },
      { name: '新低', type: 'bar', stack: 'hl', data: D.nl.map(v => -v), itemStyle: { color: t.down, opacity: .7 } },
      line('净新高 10日均', net10, t.ink, { lineStyle: { width: 1.4, color: t.ink } }),
    ],
    tooltip: Object.assign(base(t).tooltip, { valueFormatter: v => v == null ? '—' : Math.round(Math.abs(v)) }),
  }), true);

  const thr = D.thrust.map((v, i) => v ? D.adv[i] + D.dec[i] : null);
  mk('c-adv').setOption(base(t, {
    grid: { left: 8, right: 8, top: 28, bottom: 28, containLabel: true },
    xAxis: xAx(t), yAxis: [yAx(t, { scale: false }), yAx(t, { min: 0, max: 100, scale: false, position: 'right', splitLine: { show: false }, axisLabel: { color: t.muted, formatter: '{value}%' } })],
    dataZoom: zoom([0]),
    series: [
      { name: '上涨', type: 'bar', stack: 'ad', data: D.adv, itemStyle: { color: t.up, opacity: .65 }, barCategoryGap: '15%' },
      { name: '下跌', type: 'bar', stack: 'ad', data: D.dec, itemStyle: { color: t.down, opacity: .65 } },
      { name: '宽度推力', type: 'scatter', data: thr, symbol: 'triangle', symbolSize: 11, symbolOffset: [0, -8], itemStyle: { color: t.warn } },
      line('上涨量占比', ma(D.upvol.map(v => v ?? 50), 5), t.accent, { yAxisIndex: 1, lineStyle: { width: 1.4, color: t.accent } }),
    ],
  }), true);

  mk('c-msi').setOption(base(t, {
    grid: { left: 8, right: 8, top: 28, bottom: 28, containLabel: true },
    xAxis: xAx(t), yAxis: yAx(t), dataZoom: zoom([0]),
    series: [
      line('累积指数 ≥0', D.msi.map(v => v >= 0 ? v : 0), t.up, { lineStyle: { width: 0 }, areaStyle: { color: t.up, opacity: .18 }, tooltip: { show: false } }),
      line('累积指数 <0', D.msi.map(v => v < 0 ? v : 0), t.down, { lineStyle: { width: 0 }, areaStyle: { color: t.down, opacity: .18 }, tooltip: { show: false } }),
      line('累积指数', D.msi, t.ink, { markLine: { silent: true, symbol: 'none', lineStyle: { color: t.muted, type: 'dashed' }, label: { show: false }, data: [{ yAxis: 0 }] } }),
    ],
    legend: { show: false },
  }), true);

  const H = D.heat;
  mk('c-heat').setOption({
    animation: false, backgroundColor: 'transparent',
    textStyle: { color: t.muted, fontFamily: t.mono, fontSize: 11 },
    tooltip: { backgroundColor: t.panel, borderColor: t.line, textStyle: { color: t.ink, fontFamily: t.mono, fontSize: 12 },
      formatter: p => `${H.sectors[p.value[1]]}<br>${H.dates[p.value[0]]}　<b>${fmt(p.value[2])}%</b> 站上50日线` },
    grid: { left: 72, right: 16, top: 8, bottom: 64 },
    xAxis: { type: 'category', data: H.dates, axisLabel: { color: t.muted, hideOverlap: true }, axisLine: { lineStyle: { color: t.line } }, axisTick: { show: false }, splitArea: { show: false } },
    yAxis: { type: 'category', data: H.sectors, axisLabel: { color: t.ink, fontFamily: '"Noto Sans SC", sans-serif', fontSize: 12 }, axisLine: { show: false }, axisTick: { show: false } },
    visualMap: { min: 0, max: 100, calculable: false, orient: 'horizontal', left: 'center', bottom: 4, itemHeight: 220, itemWidth: 10,
      text: ['100%', '0%'], textStyle: { color: t.muted }, inRange: { color: [t.down, t.panel, t.up] } },
    series: [{ type: 'heatmap', data: H.values, itemStyle: { borderColor: t.panel, borderWidth: 1 }, emphasis: { itemStyle: { borderColor: t.ink } } }],
  }, true);

  echarts.connect(['c-ad', 'c-hl', 'c-adv', 'c-msi'].map(k => charts[k]));
}

function table() {
  const t = T();
  const c = v => v == null ? '' : `color:${v > 0 ? t.up : v < 0 ? t.down : t.muted}`;
  document.getElementById('tbl').innerHTML = `<thead><tr><th>板块</th><th>成分数</th><th>&gt;20日线</th><th>&gt;50日线</th><th>5日变化</th><th>&gt;200日线</th><th>1月中位涨幅</th></tr></thead><tbody>` +
    D.sectors.map(r => `<tr><td>${r.cn}</td><td class="num">${r.n}</td><td class="num">${fmt(r.p20)}%</td>
      <td class="num">${fmt(r.p50)}%<span class="bar" style="width:${(r.p50 || 0) * .6}px"></span></td>
      <td class="num" style="${c(r.p50_chg5)}">${sign(r.p50_chg5)}</td><td class="num">${fmt(r.p200)}%</td>
      <td class="num" style="${c(r.ret1m)}">${sign(r.ret1m)}%</td></tr>`).join('') + '</tbody>';
}

function foot() {
  const thrDates = D.dates.filter((_, i) => D.thrust[i]);
  document.getElementById('foot').innerHTML =
    `<b>口径</b>　涨跌以复权收盘价较前一日计；均线为简单移动平均，仅统计已有足够历史的股票；新高/新低为收盘价触及 252 日最高/最低。` +
    `McClellan 使用比例调整净上涨 (上涨−下跌)/(上涨+下跌)×1000。Zweig 宽度推力：上涨家数占比的 10 日 EMA 在 10 个交易日内由 40% 以下升至 61.5% 以上。` +
    `<br><b>区间内推力信号</b>　${thrDates.length ? thrDates.join('、') : '无'}。　成分股按当前名单回溯，存在幸存者偏差。本页不构成投资建议。`;
}

function renderAll() { kpis(); drawAll(); table(); foot(); }

function setRange(d) {
  range = d;
  document.querySelectorAll('#range button').forEach(b => b.setAttribute('aria-pressed', +b.dataset.d === d));
  const s = Math.max(0, N - (d || N));
  Object.entries(charts).forEach(([k, ch]) => { if (k !== 'c-heat') ch.dispatchAction({ type: 'dataZoom', startValue: s, endValue: N - 1 }); });
}
document.getElementById('range').addEventListener('click', e => { const b = e.target.closest('button'); if (b) setRange(+b.dataset.d); });
document.getElementById('conv').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  conv = b.dataset.c; try { localStorage.setItem('breadth-conv', conv); } catch (err) {}
  document.querySelectorAll('#conv button').forEach(x => x.setAttribute('aria-pressed', x === b));
  renderAll();
});
document.querySelectorAll('#conv button').forEach(x => x.setAttribute('aria-pressed', x.dataset.c === conv));

renderAll();
addEventListener('resize', () => Object.values(charts).forEach(c => c.resize()));
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', renderAll);
new MutationObserver(renderAll).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
</script>
'''


if __name__ == "__main__":
    main()
