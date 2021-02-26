"""Source code to generate the first plot in the presentation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.qarameterize import RandomCircuit

plt.style.use('phaselicious_style.mplstyle')

NUM_QUBITS = 2
PRECISION = 5
SEED = 10

rc = RandomCircuit(num_qubits=NUM_QUBITS, precision_qparams=PRECISION, seed=SEED)

max_param = 2*np.pi
binwidth = 2*np.pi/2**PRECISION

param_0_binedges = np.linspace(0-binwidth/2, 2*np.pi+binwidth/2, num=2**PRECISION+1, endpoint=True)
param_1_binedges = np.linspace(0-binwidth/2, 2*np.pi+binwidth/2, num=2**PRECISION+1, endpoint=True)
param_0_binedges, param_1_binedges = np.meshgrid(param_0_binedges, param_1_binedges)

param_0_vals_flat = np.linspace(0, 2*np.pi, num=2**PRECISION, endpoint=False)
param_1_vals_flat = np.linspace(0, 2*np.pi, num=2**PRECISION, endpoint=False)
param_0_vals, param_1_vals = np.meshgrid(param_0_vals_flat, param_1_vals_flat)

# Evaluate the cost landscape for the random circuit
score_landscape = np.zeros_like(param_0_vals)
for idx in np.ndindex(param_0_vals.shape):
    score_landscape[idx] = rc.target_prob([param_0_vals[idx], param_1_vals[idx]])


# Perform optimization on the circuit with quantum parameters
ampl_circuit, ampl_prob_func = rc.make_amplitude_amplification_circuit(num_qparams=2, K=1)
amplified_probs = ampl_prob_func(cparams=[])

probabilities = np.zeros_like(param_0_vals)
for idx in np.ndindex(param_0_vals.shape):
    probabilities[idx] = amplified_probs[idx]


roundoff = 1e-5
score_landscape[score_landscape == 0] = roundoff
probabilities[probabilities == 0] = roundoff


fig, [ax0, ax1] = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

im0 = ax0.pcolormesh(param_0_binedges, param_1_binedges, score_landscape)

divider = make_axes_locatable(ax0)
cax0 = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im0, cax=cax0)
ax0.set_aspect('equal')
ax0.set_title(r'Cost function $\langle H(\theta_0, \theta_1) \rangle$')

im1 = ax1.pcolormesh(param_0_binedges, param_1_binedges, probabilities)
divider = make_axes_locatable(ax1)
cax1 = divider.append_axes("right", size="5%", pad=0.05)

cbar1_ticks = np.linspace(0, probabilities.max(), 6, endpoint=True)
fmt = ticker.ScalarFormatter(useMathText=True)
fmt.set_powerlimits((0, 0))
cbar1 = plt.colorbar(im1, cax=cax1, format=fmt)

# cbar1.ax.set_yticklabels(["{:.1}".format(i) for i in cbar1.get_ticks()])
# cbar1.ax.set_yticklabels([f"{x:.2e}" for x in cbar1_ticks])

ax1.set_aspect('equal')
ax1.set_title('Qarameterized Circuit: \n Probability of sampling ' + r'$(\theta_0, \theta_1)$')


# Circle overlay
center_idx = np.unravel_index(score_landscape.argmax(), score_landscape.shape)
center = (param_0_vals_flat[center_idx[1]], param_1_vals_flat[center_idx[0]])
print(center)
circ = plt.Circle(center, 0.5, fill=False, color='r', lw=3)
ax1.add_artist(circ)

for ax in [ax0, ax1]:
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')


plt.savefig("img/plot1.png", dpi=200)