"""qoptimize.py - Optimization benchmarks for qarameterized circuits."""
from pennylane import numpy as np

import pennylane as qml


def optimize_steps(circuit, init_params, iterations=100, opt=None):
    """Generic optimization of a parameterized circuit using initial parameters.

    Args:
        circuit: a Pennylane circuit that accepts `params` as its argument.
            The output of this circuit should an observable that we wish to
            _maximize_.
        init_params (np.ndarray): The set of parameters to start optimization
        iterations: Number of optimization iterations to perform.
        opt: Pennylane optimizer.

    Returns:
        cost_tape (np.ndarray): Shape (iterations,) tape of cost evaluations
        param_tape (np.ndarray): Shape (iterations, len(init_params)) tape of
            parameter values during optimization

    """

    if opt is None:
        opt = qml.AdamOptimizer(stepsize=0.01)

    # Convert to minimization problem
    cost = lambda x: -1 * circuit(x)


    cost_tape = np.zeros(iterations)
    param_tape = np.zeros((iterations, len(init_params)))

    # Optimize
    params = np.copy(init_params)
    for step in range(iterations):
        params = opt.step(cost, params)
        cost_eval = cost(params)
        cost_tape[step] = cost_eval
        param_tape[step,:] = params

    return cost_tape, param_tape

