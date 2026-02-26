# ==================== RUN THIS IN GOOGLE COLAB ====================
!pip install qutip matplotlib numpy scipy --quiet

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Parameters from the thesis
r = 0.5
alpha = 1.5
cat_weight = 0.10
C0 = 0.05
N = 40

# Create states
vac = qt.basis(N, 0)
S = qt.squeeze(N, r)
squeezed = S * vac
cat = (qt.coherent(N, alpha) + qt.coherent(N, -alpha)).unit()

rho_ideal = (1 - cat_weight) * squeezed * squeezed.dag() + \
            cat_weight * cat * cat.dag()

# Wigner functions
xvec = np.linspace(-6, 6, 201)
W_ideal = qt.wigner(rho_ideal, xvec, xvec)

# Add classical noise
sigma = np.sqrt(C0)
dx = xvec[1] - xvec[0]
W_noisy = np.zeros_like(W_ideal)
for i, x in enumerate(xvec):
    for j, p in enumerate(xvec):
        gauss = np.exp( -((xvec - x)**2 + (xvec - p)**2) / (2 * sigma**2) )
        gauss /= (2 * np.pi * sigma**2)
        W_noisy[i, j] = np.sum(W_ideal * gauss) * dx**2

W_recovered = W_noisy * (1 + C0)

# Normalize
W_ideal     = W_ideal     / (np.sum(W_ideal)     * dx**2)
W_recovered = W_recovered / (np.sum(W_recovered) * dx**2)

# Plot
fig, axs = plt.subplots(1, 2, figsize=(13, 5.8))

cmap = plt.cm.RdBu
im0 = axs[0].contourf(xvec, xvec, W_ideal.T, 120, cmap=cmap, vmin=-0.35, vmax=0.35)
axs[0].set_title('Ideal State (No Classical Noise)', fontsize=14)
axs[0].set_xlabel(r'$x$', fontsize=13)
axs[0].set_ylabel(r'$p$', fontsize=13)
fig.colorbar(im0, ax=axs[0])

im1 = axs[1].contourf(xvec, xvec, W_recovered.T, 120, cmap=cmap, vmin=-0.35, vmax=0.35)
axs[1].set_title(f'After Classical Noise Subtraction ($C_0 = {C0}$)', fontsize=14)
axs[1].set_xlabel(r'$x$', fontsize=13)
axs[1].set_ylabel(r'$p$', fontsize=13)
fig.colorbar(im1, ax=axs[1])

plt.suptitle('Non-Gaussian Test State: Squeezed Vacuum ($r=0.5$) + 10% Cat Admixture ($|\\alpha|=1.5$)',
             fontsize=15.5, y=0.96)
plt.tight_layout()
plt.show()

# Save the figure (download it from Colab)
fig.savefig('non_gaussian_wigner_example.png', dpi=300, bbox_inches='tight')
print("✅ Figure saved! Download 'non_gaussian_wigner_example.png' from the left sidebar.")
