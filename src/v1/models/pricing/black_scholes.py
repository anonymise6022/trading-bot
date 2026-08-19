"""Black-Scholes fair-value pricing formula."""
import numpy as np
from scipy.stats import norm

def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """
    S: Current Stock Price
    K: Strike Price
    T: Time to Expiration in Years (e.g., 30 days = 30/365)
    r: Risk-free Interest Rate (e.g., 5% = 0.05)
    sigma: Volatility (e.g., 20% = 0.20)
    option_type: "call" or "put"
    """
    # 1. Calculate the foundational d1 and d2 components
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # 2. Compute Probability Densities
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    n_prime_d1 = norm.pdf(d1) # PDF used for Gamma, Theta, Vega
    
    if option_type.lower() == "call":
        # Call Pricing and Greeks
        price = S * N_d1 - K * np.exp(-r * T) * N_d2
        delta = N_d1 * 1
        theta = (- (S * n_prime_d1 * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N_d2) / 365
        
    elif option_type.lower() == "put":
        # Put Pricing and Greeks
        N_minus_d1 = norm.cdf(-d1)
        N_minus_d2 = norm.cdf(-d2)
        price = K * np.exp(-r * T) * N_minus_d2 - S * N_minus_d1
        delta = N_d1 - 1
        theta = (- (S * n_prime_d1 * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * N_minus_d2) / 365
    else:
        raise ValueError("option_type must be 'call' or 'put'")
        
    # Shared Greeks (Math is identical for both Calls and Puts)
    gamma = n_prime_d1 / (S * sigma * np.sqrt(T))
    vega = (S * n_prime_d1 * np.sqrt(T)) / 100 # Divided by 100 to show per 1% change in IV

    return {
        "Price": round(price, 4),
        "Delta": np.round(delta, 4),
        "Gamma": round(gamma, 4),
        "Theta": round(theta, 4), # Daily decay
        "Vega": round(vega, 4)     # Price change per 1% IV move
    }

# 🚀 Example Usage:
# Stock at $100, Strike at $100, 30 days to expiration, 5% interest rate, 20% volatility
results = black_scholes_greeks(S=100, K=100, T=30/365, r=0.05, sigma=0.20, option_type="call")
print(results)
