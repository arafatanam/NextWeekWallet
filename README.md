# NextWeekWallet

Predicts what a stock's price could look like 1 to 4 weeks from now, using its own price history. Built so someone who only understands "invest money, price goes up or down" can read the output, no finance background required.

## What this does, in plain English

1. Downloads weekly price history for a ticker you choose.
2. Builds a set of number patterns from that history (recent trend, how choppy the price has been, how far it's strayed from its average and so on).
3. Trains a few different prediction methods on those patterns.
4. Tests each method against real past weeks it never saw during training, so the accuracy numbers reflect genuine track record, not memorization.
5. Uses the best-tracking method to forecast the next 1, 2, 3 and 4 weeks, in both dollar price and "what would $1 become."
6. Shows two interactive charts so you can see the track record for yourself, not just take the numbers on faith.

## What you need before you start

- Python 3.9 or newer installed on your machine
- A free Alpha Vantage API key: https://www.alphavantage.co/support/#api-key
- Git (only if you're cloning rather than downloading a zip)

## Getting it running, step by step

**1. Get the code onto your machine**

```
git clone https://github.com/arafatanam/NextWeekWallet.git
cd NextWeekWallet
```

**2. Create a virtual environment (recommended, not required)**

This keeps NextWeekWallet's dependencies separate from anything else on your machine.

```
python -m venv venv
```

Activate it:

- Mac/Linux: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

**3. Install the dependencies**

```
pip install -r requirements.txt
```

**4. Set up your API key**

Copy the example file:

```
cp .env.example .env
```

Open `.env` in any text editor and replace `your_key_here` with the actual key you got from Alpha Vantage:

```
ALPHA_VANTAGE_API_KEY=yourrealkey
```

The `.env` file is already excluded from Git via `.gitignore`, so your key never gets committed or uploaded.

**5. Run it**

```
python next_week_wallet.py
```

You'll be prompted for a ticker symbol, e.g. `AAPL`, `CSCO`, `MSFT`. Type it in and press Enter. The first run for a given ticker downloads its full price history and caches it in a local `data/` folder, so future runs on the same ticker are faster.

## Reading the output

### Summary block

Just the basics: how much price history was found and the most recent price on record.

### The forecast

This is the part you actually want. For each of the next 1 to 4 weeks, you get one plain sentence:

> In X week(s), around DD-MM-YYYY: CSCO could be worth about $XXX.XX a share (likely between $XXX.XX and $YYY.YY).
> Every $1 invested today could become about $X. This method has gotten the direction right about X% of the time in similar past weeks.

- **"Could be worth about $X"** is the central estimate, not a guarantee. Think of it as the middle of a range of realistic outcomes.
- **"Likely between $X and $Y"** is that range, based on how far off this method's guesses have actually been in the past. A wider range means less certainty.
- **"Every $1 invested today could become about $X"** is the same estimate, scaled to $1, so you can multiply it by whatever amount you'd actually invest.
- **"Gotten the direction right about X% of the time"** tells you how often this method correctly called "up" or "down" in the past, regardless of by how much. 50% is a coin flip; 55-60% is genuinely useful for a market.

A table version of the same numbers prints underneath, in case you want to scan them quickly.

### Chart 1: actual vs. predicted return

Shows what actually happened to the stock each week, next to what each method would have predicted, over the last 2 years. Everything is hidden by default except the actual result, the best-tracking method and a "naive baseline" (a method that just assumes the recent trend continues, included as a sanity check). Hover anywhere on the chart to see every visible line's value at that date. Click a name in the legend to add or remove it.

### Chart 2: growth of $1

Shows what $1 would have grown into if you'd followed each method's up/down signal each week, versus just buying and holding the whole time. Same 2-year window, same hover and click-to-toggle behavior.

### Advanced detail (optional)

A full comparison table per horizon, with the raw accuracy numbers for every method. You don't need this section to use the tool, it's there for anyone curious about the mechanics. Glossary below.

## Glossary

**Ridge**: A simple, cautious method that fits a straight-line relationship between the patterns and next week's return. Doesn't overreact to noise.

**Random Forest**: Builds hundreds of small decision trees on random slices of the data and averages their votes. Good at catching non-obvious combinations of patterns.

**HistGradientBoosting**: Builds decision trees one after another, where each new tree focuses on correcting the mistakes of the ones before it.

**XGBoost**: A faster, more heavily tuned version of the same boosting idea, widely used in real-world forecasting competitions.

**Ensemble**: The average of all four methods' guesses. Usually more reliable than any single method, since it smooths out each one's individual blind spots.

**Naive baseline**: The simplest possible guess: "the recent trend continues." It's a sanity check. If the real methods can't beat this, they're not adding value.

**RMSE (Root Mean Squared Error)**: On average, how far off a method's guesses were. Lower is better.

**MAE (Mean Absolute Error)**: Similar to RMSE, but treats every miss equally instead of punishing big misses harder. Lower is better.

**R² (R-squared)**: How much of the week-to-week movement a method explains, from 0 (explains nothing) to 1 (explains everything). For weekly stock returns, values close to 0 are normal, not a bug.

**DirAcc (Directional Accuracy)**: How often a method got the direction right (up or down), regardless of the size of the move. 50% is a coin flip.

**IC (Information Coefficient)**: How well a method's ranking of "better vs. worse weeks" lines up with what actually happened. Closer to 1 or -1 is stronger; 0 means no relationship.

## Why the accuracy numbers look modest

Weekly stock returns are close to random. A method that calls the direction right 53-55% of the time is doing real, usable work, that edge compounds. If a method here ever shows 70%+ accuracy, treat it with suspicion rather than excitement, it usually means the backtest is leaking future information rather than the method being unusually good.

## Troubleshooting

- **"No Alpha Vantage key found"**: you haven't created `.env` yet, or it's missing the key. Repeat step 4 above.
- **"Alpha Vantage did not return price data for TICKER"**: either the ticker doesn't exist, or you've hit Alpha Vantage's free-tier rate limit (5 requests/minute, 25/day on the free plan). Wait a bit and try again.
- **Charts don't open**: Plotly opens charts in your default web browser when run outside a notebook. If nothing opens automatically, check your terminal output for a local file path or URL to open manually.

## Disclaimer

This tool produces statistical estimates based on historical patterns. It is not financial advice and past performance does not guarantee future results.

## Project structure

```
NextWeekWallet/
├── README.md
├── requirements.txt
├── next_week_wallet.py
├── .env.example
├── .env               (you create this, never committed)
├── .gitignore
└── data/               (auto-created, cached price history, never committed)
```
