"""
Wigner function comparison illustrating coherence requirements for a valid Local Oscillator
Used in Section 3.2.3 of the Master's thesis:
"Balanced Homodyne Detection without Coherent State Local Oscillator"
by Dhrithi Maria, Paderborn University, February 2026

Compares:
- Invalid LO: Pure Fock state |1⟩
- Valid LO: Coherent superposition (|0⟩ + |1⟩)/√2 (showing interference fringes)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
from matplotlib import rcParams
import math

# Professional style
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12

def wigner_laguerre(x, p, rho):
    """
    Computes the Wigner function for a density matrix rho in the Fock basis (N=2 here).
    rho is a 2D complex array where rho[m, n] = <m|rho|n>.
    """
    alpha = x + 1j * p
    alpha_sq = np.abs(alpha)**2
    W = np.zeros_like(x, dtype=float)
    N = rho.shape[0]

    for m in range(N):
        for n in range(m + 1):
            if np.abs(rho[m, n]) > 1e-10:
                diff = m - n
                coeff = (2.0 / np.pi) * (-1)**n * np.sqrt(math.factorial(n) / math.factorial(m))
                complex_part = (2 * np.conjugate(alpha))**diff
                exp_part = np.exp(-2 * alpha_sq)
                laguerre_part = genlaguerre(n, diff)(4 * alpha_sq)
                term = coeff * complex_part * exp_part * laguerre_part * rho[m, n]

                if m == n:
                    W += np.real(term)
                else:
                    W += 2 * np.real(term)  # Hermitian symmetry

    return W

# Grid
x = np.linspace(-4, 4, 300)
p = np.linspace(-4, 4, 300)
X, P = np.meshgrid(x, p)

# State A: Invalid LO — Pure Fock |1⟩
rho_fock = np.zeros((3, 3), dtype=complex)
rho_fock[1, 1] = 1.0

# State B: Valid LO — Normalized superposition (|0⟩ + |1⟩)/√2
rho_super = np.zeros((3, 3), dtype=complex)
rho_super[0, 0] = 0.5
rho_super[1, 1] = 0.5
rho_super[0, 1] = 0.5
rho_super[1, 0] = 0.5

# Compute Wigner functions
W_fock = wigner_laguerre(X, P, rho_fock)
W_super = wigner_laguerre(X, P, rho_super)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cmap = 'RdBu_r'

# Invalid LO
c1 = axes[0].pcolormesh(X, P, W_fock, cmap=cmap, shading='auto', vmin=-0.4, vmax=0.4)
axes[0].set_title(r'(a) Invalid LO: Pure Fock State $|1\rangle$', fontsize=14)
axes[0].set_xlabel(r'$x$ (Position Quadrature)', fontsize=12)
axes[0].set_ylabel(r'$p$ (Momentum Quadrature)', fontsize=12)
axes[0].set_aspect('equal')
axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axes[0].axvline(0, color='gray', linestyle='--', linewidth=0.8)

# Valid LO
c2 = axes[1].pcolormesh(X, P, W_super, cmap=cmap, shading='auto', vmin=-0.4, vmax=0.4)
axes[1].set_title(r'(b) Valid LO: Superposition $\frac{|0\rangle + |1\rangle}{\sqrt{2}}$', fontsize=14)
axes[1].set_xlabel(r'$x$ (Position Quadrature)', fontsize=12)
axes[1].set_yticks([])
axes[1].set_aspect('equal')
axes[1].axhline(0, color='gray', linestyle='--', linewidth=0.8)
axes[1].axvline(0, color='gray', linestyle='--', linewidth=0.8)

# Shared colorbar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
cbar = fig.colorbar(c1, cax=cbar_ax)
cbar.set_label(r'$W(x,p)$', rotation=270, labelpad=15)

plt.suptitle('Coherence Requirements for a Valid Local Oscillator\n(Section 3.2.3)', fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('wigner_coherence_requirements.jpg', dpi=400, bbox_inches='tight')
plt.close()
print("✅ Generated: wigner_coherence_requirements.jpg")
