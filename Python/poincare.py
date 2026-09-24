import numpy as np
import matplotlib.pyplot as plt

def cayley(z):
    return (z - 1j) / (z + 1j)

def poincare_to_klein(w):
    r2 = np.abs(w)**2
    return 2 * w / (1 + r2)

theta = np.linspace(0, np.pi, 800)

# Geodesics in the half-plane
z_l = 5 * np.exp(1j * theta)
y_k = np.logspace(-3, 3, 800)
z_k = 1j * y_k
z_m = -2.2 + 2.2 * np.exp(1j * theta)
z_n = 2.5 + 1.5 * np.exp(1j * theta)

# Transform to Poincaré disk
w_l, w_k, w_m, w_n = cayley(z_l), cayley(z_k), cayley(z_m), cayley(z_n)

# Transform to Klein-Beltrami
k_l, k_k, k_m, k_n = (poincare_to_klein(w) for w in (w_l, w_k, w_m, w_n))

# Representative points for labels (in the half-plane)
z_lab = {
    'l': 5j,              # top of big semicircle
    'k': 2j,              # point on vertical ray
    'm': -2.2 + 2.2j,     # top of left semicircle
    'n': 2.5 + 1.5j,      # top of right semicircle
}

# Transformed label positions
w_lab = {k: cayley(z) for k, z in z_lab.items()}
k_lab = {k: poincare_to_klein(w) for k, w in w_lab.items()}

# Small outward offsets so the label sits just off the curve
w_off = {'l': 0.10j, 'k': 0.10j, 'm': 0.10 + 0.05j, 'n': 0.08 - 0.05j}
k_off = {'l': 0.06j, 'k': 0.08j, 'm': 0.05 + 0.05j, 'n': 0.05 - 0.05j}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors = {'l': 'blue', 'k': 'red', 'm': 'green', 'n': 'darkorange'}

# --- 1. Half-plane ---
ax = axes[0]
ax.plot(z_l.real, z_l.imag, color=colors['l'], linewidth=2)
ax.plot(z_k.real, z_k.imag, color=colors['k'], linewidth=2)
ax.plot(z_m.real, z_m.imag, color=colors['m'], linewidth=2)
ax.plot(z_n.real, z_n.imag, color=colors['n'], linewidth=2)
ax.axhline(0, color='gray', linestyle='--', linewidth=1)
ax.set_xlim(-6, 6); ax.set_ylim(-0.5, 6.5)
ax.set_aspect('equal')
ax.set_title('Poincaré Half-Plane Model', fontsize=13)
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.text(-2.5, 4.8, 'l', fontsize=16, ha='center', color=colors['l'])
ax.text(0.15, 3.0, 'k', fontsize=16, ha='left', color=colors['k'])
ax.text(-2.2, 2.4, 'm', fontsize=16, ha='center', color=colors['m'])
ax.text(2.5, 1.7, 'n', fontsize=16, ha='center', color=colors['n'])

# --- 2. Poincaré disk ---
ax = axes[1]
ax.plot(w_l.real, w_l.imag, color=colors['l'], linewidth=2)
ax.plot(w_k.real, w_k.imag, color=colors['k'], linewidth=2)
ax.plot(w_m.real, w_m.imag, color=colors['m'], linewidth=2)
ax.plot(w_n.real, w_n.imag, color=colors['n'], linewidth=2)
ax.add_patch(plt.Circle((0, 0), 1, fill=False, color='black', linewidth=1))
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.set_title('Poincaré Disk Model', fontsize=13)
ax.axhline(0, color='gray', linestyle=':', linewidth=0.6)
ax.axvline(0, color='gray', linestyle=':', linewidth=0.6)
for key in 'lkmn':
    p = w_lab[key] + w_off[key]
    ax.text(p.real, p.imag, key, fontsize=16, ha='center',
            color=colors[key])

# --- 3. Klein-Beltrami disk ---
ax = axes[2]
ax.plot(k_l.real, k_l.imag, color=colors['l'], linewidth=2)
ax.plot(k_k.real, k_k.imag, color=colors['k'], linewidth=2)
ax.plot(k_m.real, k_m.imag, color=colors['m'], linewidth=2)
ax.plot(k_n.real, k_n.imag, color=colors['n'], linewidth=2)
ax.add_patch(plt.Circle((0, 0), 1, fill=False, color='black', linewidth=1))
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.set_title('Klein-Beltrami Model', fontsize=13)
ax.axhline(0, color='gray', linestyle=':', linewidth=0.6)
ax.axvline(0, color='gray', linestyle=':', linewidth=0.6)
for key in 'lkmn':
    p = k_lab[key] + k_off[key]
    ax.text(p.real, p.imag, key, fontsize=16, ha='center',
            color=colors[key])

plt.tight_layout()
plt.show()
