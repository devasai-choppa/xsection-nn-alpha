# Narration

Voice: `en-US-AndrewNeural`. No background music.
The full demo is about two minutes. The short cut is 45 seconds and vertical.

## title

Each month, this model ranks about five hundred large-cap stocks and bets on the gap between the top and the bottom.

## data

Only information known at month-end goes in. Momentum, reversal, volatility, liquidity, beta, and the largest daily move. Each feature is rank-normalized, that month, from minus one to one.

## network

Twelve inputs, three hidden layers, one score. Brightness is a real activation. Edge color is the sign of a real weight. A top name, a middle name, and a bottom name, from December 2024.

## walk

Test months are never used for tuning. Once a year the model is refit on an expanding history, with a one-month embargo before the next test block.

## dashboard

The dashboard replays every test month. Move the date, pick a ticker, and the diagram is that stock's forward pass. Integrated gradients attribute the score to the twelve features. The cost slider does not refit the model. It reprices turnover that was already recorded. Ten basis points per side is the published default.

## results

From 2012 through September 2026, the network made about ten percent a year after costs, Sharpe near 0.8. LightGBM was a little stronger. Momentum plus reversal did not survive the costs.

## end

Today's S&P 500 list embeds survivorship bias, and the prices are free data. This is research code, not a trading system. github.com/TechieGoku2623.

## short_title

A neural net ranks large-cap stocks. Long the top decile, short the bottom.

## short_network

Real weights. Real activations. One score per stock.

## short_dashboard

Every test month, with live costs and feature attributions.

## short_results

After ten basis points, the network's Sharpe is about 0.8. A tree model edges it.

## short_end

Survivorship bias included. Not a trading system. github.com/TechieGoku2623.
