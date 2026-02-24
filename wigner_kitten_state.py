"""
Wigner function visualization for Schrödinger kitten / cat state
Used in Section 4.3.2 of the Master's thesis:
"Balanced Homodyne Detection without Coherent State Local Oscillator"
by Dhrithi Maria, Paderborn University, February 2026

This script generates the exact figure shown in the thesis (even cat state approximation
of the |0⟩ + |2⟩ kitten state, illustrating interference fringes).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Professional style
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12

def plot_wigner(x, p, W, title, filename):
    plt.figure(figsize=(6.5, 5.2))
    contour = plt.contourf(x, p, W, levels=60, cmap='RdBu_r', extend='both')
    cbar = plt.colorbar(contour)
    cbar.set_label(r'$W(x,p)$', rotation=270, labelpad=20)
    
    plt.xlabel(r'$x$ (Position Quadrature)')
    plt.ylabel(r'$p$ (Momentum Quadrature)')
    plt.title(title)
    plt.axhline(0, color='black', linestyle='--', alpha=0.4, linewidth=0.8)
    plt.axvline(0, color='black', linestyle='--', alpha=0.4, linewidth=0.8)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(filename, dpi=400, bbox_inches='tight')
    plt.close()
    print(f"✅ Generated: {filename}")

# Grid
x = np.linspace(-4, 4, 300)
p = np.linspace(-4, 4, 300)
X, P = np.meshgrid(x, p)

# Even cat state (α = 2) — visually equivalent to normalized |0⟩ + |2⟩ kitten for illustration
alpha = 2.0
R2 = X**2 + P**2

# Unnormalized Wigner function for |α⟩ + |-α⟩ (standard textbook form)
W_cat = (np.exp(-(X - alpha)**2 - P**2) +
         np.exp(-(X + alpha)**2 - P**2) +
         2 * np.exp(-R2) * np.cos(4 * alpha * P))

plot_wigner(x, p, W_cat,
            'Schrödinger Kitten State (Even Cat Approximation)\n(Used for visualization in Section 4.3.2)',
            'wigner_kitten.jpg')
