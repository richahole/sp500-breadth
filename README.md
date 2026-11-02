# 标普500市场宽度

每个美股交易日收盘后，GitHub Actions 自动下载标普500成分股日线，计算市场宽度指标，并发布为可交互网页（GitHub Pages）。

**指标**：站上 20/50/200 日线比例、腾落线、McClellan 振荡器与累积指数、52 周新高/新低、涨跌家数与上涨成交量占比、Zweig 宽度推力、板块 50 日线宽度热力图与快照表。

**数据**：Yahoo Finance 复权日线（yfinance）；成分股名单来自 [datasets/s-and-p-500-companies](https://github.com/datasets/s-and-p-500-companies)。每日指标历史保存在 `data/breadth_daily.csv`。成分股按当前名单回溯，存在幸存者偏差。不构成投资建议。

## 本地运行

```bash
pip install -r requirements.txt
python breadth.py              # 生成 sp500_breadth.html
python breadth.py --selftest   # 随机数据测试页面
```

## 自动更新

`.github/workflows/update.yml` 在 UTC 22:30（周一至周五）运行，也可在 Actions 页面点 *Run workflow* 手动触发。首次使用需在 Settings → Pages → Source 选择 **GitHub Actions**。
