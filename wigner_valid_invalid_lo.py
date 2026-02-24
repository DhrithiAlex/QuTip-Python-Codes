"""
Wigner function comparison: Valid LO (coherent state) vs Invalid LO (Fock state)
Used in Section 3.2.3 of the Master's thesis:
"Balanced Homodyne Detection without Coherent State Local Oscillator"
by Dhrithi Maria, Paderborn University, February 2026

This script reproduces Figure 3.2 showing the adjacency condition.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import laguerre
from matplotlib import rcParams

# Professional style
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12

def wigner_coherent(x, p, alpha):
    x0 = np.sqrt(2) * alpha.real
    p0 = np.sqrt(2) * alpha.imag
    return (1.0 / np.pi) * np.exp(-(x - x0)**2 - (p - p0)**2)

def wigner_fock(x, p, n):
    r2 = x**2 + p**2
    return (1.0 / np.pi) * ((-1)**n) * np.exp(-r2) * laguerre(n)(2 * r2)

def plot_wigner_comparison():
    # Grid
    x_vec = np.linspace(-4, 4, 400)
    p_vec = np.linspace(-4, 4, 400)
    X, P = np.meshgrid(x_vec, p_vec)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Case A: Valid LO — Coherent state |α=1.5⟩
    im1 = ax1.contourf(X, P, wigner_coherent(X, P, 1.5 + 0j), 100, cmap='RdBu_r')
    ax1.set_title('Case A: Valid LO (Coherent State)')
    ax1.set_xlabel(r'$x$ (Position Quadrature)')
    ax1.set_ylabel(r'$p$ (Momentum Quadrature)')
    fig.colorbar(im1, ax=ax1)

    # Case B: Invalid LO — Pure Fock state |n=2⟩
    im2 = ax2.contourf(X, P, wigner_fock(X, P, 2), 100, cmap='RdBu_r')
    ax2.set_title('Case B: Invalid LO (Fock State)')
    ax2.set_xlabel(r'$x$ (Position Quadrature)')
    ax2.set_ylabel(r'$p$ (Momentum Quadrature)')
    fig.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig('wigner_valid_invalid_lo.jpg', dpi=400, bbox_inches='tight')
    plt.close()
    print("✅ Generated: wigner_valid_invalid_lo.jpg")

if __name__ == "__main__":
    plot_wigner_comparison()
