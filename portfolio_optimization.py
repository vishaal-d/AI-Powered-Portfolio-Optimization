import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define Indian stocks from different market caps
tickers = ["RELIANCE.NS", "INFY.NS", "HDFCBANK.NS",  # Large Cap
           "TATAPOWER.NS", "VOLTAS.NS", "DEEPAKNTR.NS",  # Mid Cap
           "IEX.NS", "ROUTE.NS", "KNRCON.NS", "DIXON.NS"]  # Small Cap

# Define stock categories for visualization
categories = ["Large Cap"] * 3 + ["Mid Cap"] * 3 + ["Small Cap"] * 4

# Define date range
start_date = "2023-01-01"
end_date = "2024-01-01"

# Download stock data from Yahoo Finance
data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)

# Use Adjusted Close price if available
if "Adj Close" in data.columns:
    data = data["Adj Close"]
else:
    data = data["Close"]

# Calculate daily returns
returns = data.pct_change().dropna()

# Portfolio optimization function
def portfolio_performance(weights, returns):
    portfolio_return = np.sum(weights * returns.mean()) * 252  # Annualized return
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualized volatility
    sharpe_ratio = portfolio_return / portfolio_volatility  # Risk-adjusted return
    return portfolio_return, portfolio_volatility, sharpe_ratio

# Objective function: Minimize negative Sharpe ratio
def negative_sharpe(weights, returns):
    return -portfolio_performance(weights, returns)[2]  # Minimize negative Sharpe ratio

# Constraints: Weights sum to 1
constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

# Bounds for weights (between 0% and 100%)
bounds = [(0, 1) for _ in range(len(tickers))]

# Initial equal weights
initial_weights = np.ones(len(tickers)) / len(tickers)

# Optimization
optimized = minimize(negative_sharpe, initial_weights, args=(returns,), method="SLSQP", bounds=bounds, constraints=constraints)
optimal_weights = optimized.x

# Get optimized portfolio performance
opt_return, opt_volatility, opt_sharpe = portfolio_performance(optimal_weights, returns)

# Print results
print("\n🔹 Optimized Portfolio Allocation:")
for stock, weight in zip(tickers, optimal_weights):
    print(f"{stock}: {weight:.2%}")
print(f"\n📈 Expected Annual Return: {opt_return:.2%}")
print(f"📉 Expected Volatility (Risk): {opt_volatility:.2%}")
print(f"⚡ Sharpe Ratio: {opt_sharpe:.2f}")

# 📊 **Visualizations**

# 1️⃣ Pie Chart → Portfolio Allocation
# Filter stocks with allocation >= 1%
non_small_weights = [(stock, weight) for stock, weight in zip(tickers, optimal_weights) if weight >= 0.01]
filtered_stocks, filtered_weights = zip(*non_small_weights) if non_small_weights else ([], [])

# Plot Pie Chart → Portfolio Allocation (Only Stocks with ≥1% Allocation)
plt.figure(figsize=(7, 7))
plt.pie(filtered_weights, labels=filtered_stocks, autopct="%1.1f%%", colors=plt.cm.Paired.colors)
plt.title("Optimized Portfolio Allocation")
plt.show()



# 2️⃣ Bar Chart → Stock-wise Allocation
plt.figure(figsize=(7, 5))
plt.bar(tickers, optimal_weights, color=plt.cm.Paired.colors)
plt.xlabel("Stocks")
plt.ylabel("Weight in Portfolio")
plt.title("Stock-wise Portfolio Allocation")
plt.xticks(rotation=45)
plt.show()

# 3️⃣ Histogram → Distribution of Daily Returns (Risk Visualization)
plt.figure(figsize=(8, 5))
plt.hist(returns, bins=50, alpha=0.6, label=tickers)
plt.xlabel("Daily Returns")
plt.ylabel("Frequency")
plt.title("Distribution of Stock Returns (Risk Visualization)")
plt.legend()
plt.show()

# 4️⃣ Line Graph → Cumulative Portfolio Returns
cumulative_returns = (returns + 1).cumprod()
plt.figure(figsize=(10, 5))
for stock in tickers:
    plt.plot(cumulative_returns[stock], label=stock)
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.title("Stock-wise Cumulative Returns")
plt.legend()
plt.show()

# 5️⃣ Bar Chart → Volatility of Each Stock
volatility = returns.std() * np.sqrt(252)  # Annualized volatility
plt.figure(figsize=(8, 5))
plt.bar(tickers, volatility, color="red")
plt.xlabel("Stocks")
plt.ylabel("Annualized Volatility")
plt.title("Stock Volatility (Risk Indicator)")
plt.xticks(rotation=45)
plt.show()

# 6️⃣ Pie Chart → Market Cap Allocation
category_weights = {}
for category, weight in zip(categories, optimal_weights):
    category_weights[category] = category_weights.get(category, 0) + weight

plt.figure(figsize=(7, 7))
plt.pie(category_weights.values(), labels=category_weights.keys(), autopct="%1.1f%%", colors=["blue", "orange", "green"])
plt.title("Market Cap Allocation (Large, Mid, Small Cap)")
plt.show()
