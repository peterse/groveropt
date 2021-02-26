import numpy as np
import matplotlib.pyplot as plt

from src import io

plt.style.use('phaselicious_style.mplstyle')

def moving_average(x, width=50):
    return np.convolve(x, np.ones(width)/width, mode='valid')

fig, axes = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
axes = [axes]
# Plot the grover-assisted optimization
cost_tape_assisted = np.load(io.get_quantum_cost_tape_fname())
param_tape_assisted = np.load(io.get_quantum_param_tape_fname())

y1 = moving_average(cost_tape_assisted)
patch1, = axes[0].plot(range(len(y1)), y1, c='g', label='Qarameterized')
# axes[1].scatter(param_tape_assisted[:,0], param_tape_assisted[:,1], c='g', s=5)

# plot the classical optimizations
n_trials = 5
for k in range(n_trials):
    cost_tape = np.load(io.get_classical_cost_tape_fname(k))
    param_tape = np.load(io.get_classical_param_tape_fname(k))
    y0 = moving_average(cost_tape)
    patch2, = axes[0].plot(range(len(y0)), y0, c='r', alpha=0.3, label='Random initialization')
    # axes[1].scatter(param_tape[:,0], param_tape[:,1], c='r', s=5)

first = axes[0].legend(handles = [patch1, patch2],
                      loc=(.67, 0.93),
                      # prop={'size': 'large'},
                      ncol=1,
                      # bbox_to_anchor=(0.4, .2, .8, 0.3)
                      framealpha=1,
                      )

axes[0].set_xlabel("Iterations")
axes[0].set_ylabel(r"Loss value for $\langle H(\theta) \rangle$")

plt.savefig("img/plot2.png", dpi=300)
plt.savefig("img/plot2_src.svg")

plt.show()