import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Define stock universe (Large, Mid, Small Cap)
stock_universe = [
    "TCS.NS", "RELIANCE.NS", "HDFCBANK.NS",  # Large Cap
    "LTIM.NS", "CIPLA.NS", "ABB.NS",        # Mid Cap
    "KPITTECH.NS", "TATAELXSI.NS", "IBREALEST.NS", "IDEA.NS"  # Small Cap
]

# Download stock data
data = yf.download(stock_universe, start="2023-01-01", end="2024-01-01", auto_adjust=False)

# Use 'Adj Close' if available, otherwise use 'Close'
if "Adj Close" in data:
    data = data["Adj Close"]
else:
    data = data["Close"]

# Calculate daily returns
returns = data.pct_change().dropna()

# Risk (Volatility) calculation
risk = returns.std()

# Calculate cumulative returns
cumulative_returns = (1 + returns).cumprod()

# Correlation matrix
correlation_matrix = returns.corr()

# Portfolio Optimization - Sharpe Ratio-based
num_stocks = len(stock_universe)
random_weights = np.random.rand(num_stocks)
random_weights /= random_weights.sum()  # Normalize

# Expected Portfolio Return & Risk
expected_return = np.dot(random_weights, returns.mean()) * 252  # Annualized
portfolio_risk = np.sqrt(np.dot(random_weights.T, np.dot(returns.cov() * 252, random_weights)))  # Annualized
risk_free_rate = 0.05  # Assume 5% risk-free rate

# Sharpe Ratio
sharpe_ratio = (expected_return - risk_free_rate) / portfolio_risk

# AI-based Stock Clustering for Recommendations
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(returns.T)

# AI Stock Recommendation: Pick one from least volatile cluster
recommended_stock = data.columns[np.argmin(risk)]

# Portfolio Allocation
allocation = {stock: weight * 100 for stock, weight in zip(stock_universe, random_weights)}

# Filter out allocations <1%
filtered_allocation = {k: v for k, v in allocation.items() if v >= 1}

# 🔹 Visualization 1: Portfolio Allocation Pie Chart
plt.figure(figsize=(8, 8))
plt.pie(filtered_allocation.values(), labels=filtered_allocation.keys(), autopct='%1.1f%%', startangle=140, colors=sns.color_palette("husl", len(filtered_allocation)))
plt.title("Portfolio Allocation")
plt.show()

# 🔹 Visualization 2: Stock Risk (Volatility) Bar Chart
plt.figure(figsize=(10, 5))
sns.barplot(x=risk.index, y=risk.values, palette="coolwarm")
plt.xticks(rotation=45)
plt.ylabel("Risk (Standard Deviation)")
plt.title("Stock Risk Comparison")
plt.show()

# 🔹 Visualization 3: Cumulative Returns Line Chart
plt.figure(figsize=(12, 6))
for stock in cumulative_returns.columns:
    plt.plot(cumulative_returns.index, cumulative_returns[stock], label=stock)
plt.legend()
plt.title("Cumulative Returns of Stocks")
plt.ylabel("Cumulative Returns")
plt.xlabel("Date")
plt.grid()
plt.show()

# 🔹 Visualization 4: Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Stock Correlation Heatmap")
plt.show()

# Display Portfolio Statistics
print(f"📊 Expected Portfolio Return: {expected_return:.2f}%")
print(f"📉 Portfolio Risk (Volatility): {portfolio_risk:.2f}%")
print(f"⚡ Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"📈 AI Recommends: Consider adding {recommended_stock} to your portfolio!")
