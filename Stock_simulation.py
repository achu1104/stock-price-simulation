
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Prepare the tools
rng = np.random.default_rng(42)
TRADING_DAYS = 252

# 2. Price simulator using geometric Brownian motion
def simulate_prices(start_price, annual_return, annual_volatility,
                    days, simulations=1, rng=None):

    daily_return = annual_return / TRADING_DAYS
    daily_volatility = annual_volatility / np.sqrt(TRADING_DAYS)

    shocks = rng.normal(
        0, 1, size=(days, simulations)
    )

    log_returns = (
        daily_return - 0.5 * daily_volatility**2
        + daily_volatility * shocks
    )

    prices = start_price * np.exp(
        np.vstack([
            np.zeros(simulations),
            np.cumsum(log_returns, axis=0)
        ])
    )

    return prices


# 3. Create historical price data
history_days = 504
dates = pd.bdate_range(end="2025-12-31", periods=history_days + 1)

historical_prices = simulate_prices(
    start_price=100,
    annual_return=0.10,
    annual_volatility=0.25,
    days=history_days,
    rng=rng
)[:, 0]

df = pd.DataFrame(
    {"Close": historical_prices},
    index=dates
)

# 4. Calculate moving averages
df["SMA 20"] = df["Close"].rolling(window=20).mean()
df["SMA 50"] = df["Close"].rolling(window=50).mean()

# 5. Estimate annual return and volatility
daily_returns = df["Close"].pct_change().dropna()

annual_return = daily_returns.mean() * TRADING_DAYS
annual_volatility = daily_returns.std() * np.sqrt(TRADING_DAYS)

current_price = df["Close"].iloc[-1]

print("Current price:", round(current_price, 2))
print("Estimated annual return:", round(annual_return * 100, 2), "%")
print("Estimated annual volatility:", round(annual_volatility * 100, 2), "%")

# 6. Forecast 1,000 possible price paths
forecast_days = 252
simulations = 1000

forecast = simulate_prices(
    start_price=current_price,
    annual_return=annual_return,
    annual_volatility=annual_volatility,
    days=forecast_days,
    simulations=simulations,
    rng=rng
)

forecast_dates = pd.bdate_range(
    start=df.index[-1],
    periods=forecast_days + 1
)

# 7. Calculate risk measures
final_prices = forecast[-1]

p5 = np.percentile(final_prices, 5)
median_price = np.percentile(final_prices, 50)
p95 = np.percentile(final_prices, 95)

var_95 = current_price - p5
probability_above = np.mean(final_prices > current_price) * 100

print("\n--- One-Year Forecast ---")
print("Median forecast:", round(median_price, 2))
print("5th percentile:", round(p5, 2))
print("95th percentile:", round(p95, 2))
print("95% Value-at-Risk:", round(var_95, 2))
print("Probability of finishing above today's price:",
      round(probability_above, 1), "%")

# 8. Draw the fan chart
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]

ax.plot(df.index, df["Close"], color="black",
        linewidth=1, label="Close")
ax.plot(df.index, df["SMA 20"], label="SMA 20",
        alpha=0.8)
ax.plot(df.index, df["SMA 50"], label="SMA 50",
        alpha=0.8)

# Show 100 sample forecast paths
sample_indices = rng.choice(simulations, size=100, replace=False)

for i in sample_indices:
    ax.plot(forecast_dates, forecast[:, i],
            color="cornflowerblue", alpha=0.08,
            linewidth=0.7)

# Calculate the 5th, 50th and 95th percentile paths
lower_band = np.percentile(forecast, 5, axis=1)
median_path = np.percentile(forecast, 50, axis=1)
upper_band = np.percentile(forecast, 95, axis=1)

ax.fill_between(
    forecast_dates, lower_band, upper_band,
    color="green", alpha=0.15, label="5-95% band"
)

ax.plot(forecast_dates, median_path,
        color="green", linewidth=2, label="Forecast median")

ax.set_title("Price History and 1-Year Monte Carlo Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("USD")
ax.legend()
ax.grid(alpha=0.2)

# 9. Draw the final-price histogram
ax2 = axes[1]

ax2.hist(final_prices, bins=35, color="cornflowerblue",
         edgecolor="white")

ax2.axvline(current_price, color="black",
            linestyle="--", label="Today")
ax2.axvline(p5, color="red",
            linestyle=":", label="5th percentile")

ax2.set_title("Distribution of Final Prices")
ax2.set_xlabel("USD")
ax2.set_ylabel("Number of Simulations")
ax2.legend()
ax2.grid(alpha=0.2)

# 10. Save the chart
plt.tight_layout()
plt.savefig("stock_forecast.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nChart saved as stock_forecast.png")
