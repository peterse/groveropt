import qarameterize
import qoptimize
import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np

NUM_QUBITS = 3
NUM_CPARAMS = 2
PRECISION = 5
SEED = 11
dev = qml.device('default.qubit', wires=range(NUM_QUBITS), shots=1000, analytic=False)

version = 2


def moving_average(x, width):
    return np.convolve(x, np.ones(width)/width, mode='valid')


if version == 1:
    rc = qarameterize.LayerwiseRandomCircuit(num_qubits=NUM_QUBITS, num_params=NUM_CPARAMS, precision_qparams=PRECISION, seed=SEED)

    def get_circuit(params):
        rc.rand_circuit(params, num_qparams=0)
        H = np.zeros((2 ** NUM_QUBITS, 2 ** NUM_QUBITS))
        H[0, 0] = 1
        wirelist = [i for i in range(NUM_QUBITS)]
        return qml.expval(qml.Hermitian(H, wirelist))
        # return qml.expval(qml.PauliZ(0))

    # circuit = get_circuit
    classical_circuit = qml.QNode(get_circuit, dev)
    # classical_circuit(cparams)
    #
    # with open('circuit_log.txt', 'w') as f:
    #     print("Classical")
    #     print(classical_circuit.draw(), file=f)
    # #

    # Optimize the prepared circuit
    initial_params = np.random.random(NUM_CPARAMS) * np.pi
    n_iter = 1500
    opt = qml.AdamOptimizer(stepsize=0.003)
    cost_tape, param_tape = qoptimize.optimize_steps(classical_circuit, initial_params, iterations=n_iter, opt=opt)

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))

    vals = moving_average(cost_tape, 20)
    axes[0].plot(range(len(vals)), vals, c='r')
    axes[1].scatter(param_tape[:,0], param_tape[:,1], c='b', s=5)
    plt.show()

elif version == 2:
    #
    # # Force that total number of params equals number of qparams to push out any
    # # classical param inputs
    NUM_QPARAMS = 2 # all params quantum
    rc = qarameterize.LayerwiseRandomCircuit(num_qubits=NUM_QUBITS, num_params= NUM_QPARAMS, precision_qparams=PRECISION, seed=SEED)
    ampl_circuit, ampl_prob_func = rc.make_amplitude_amplification_circuit(num_qparams=2, K=1)
    amplified_probs = ampl_prob_func(cparams=[])

    param_vals = np.linspace(0, 2*np.pi, num=2**PRECISION, endpoint=False)
    shape = (len(param_vals), len(param_vals))

    # Create a grid of probabilities
    probabilities = np.zeros(shape)
    for idx in np.ndindex(shape):
        probabilities[idx] = amplified_probs[idx]
    p_flat = probabilities.flatten()

    # Sample from the grid
    nsamples = 100
    idx_samples = np.random.choice(np.arange(len(p_flat)), p=p_flat, size=nsamples)
    idx_samples = [np.unravel_index(x, shape) for x in idx_samples]
    theta_samples = []
    # Generate the lists of parameters associated with every sample
    for idx in idx_samples:
        theta_samples.append([param_vals[i] for i in idx])


    # COMPARE OPTIMIZATIONS
    rc = qarameterize.LayerwiseRandomCircuit(num_qubits=NUM_QUBITS, num_params=NUM_CPARAMS, precision_qparams=PRECISION, seed=SEED)

    def get_circuit(params):
        rc.rand_circuit(params, num_qparams=0)
        H = np.zeros((2 ** NUM_QUBITS, 2 ** NUM_QUBITS))
        H[0, 0] = 1
        wirelist = [i for i in range(NUM_QUBITS)]
        return qml.expval(qml.Hermitian(H, wirelist))
        # return qml.expval(qml.PauliZ(0))

    # circuit = get_circuit
    classical_circuit = qml.QNode(get_circuit, dev)

    n_iter = 300

    assisted_start = np.asarray(theta_samples).mean(axis=0)
    opt = qml.AdamOptimizer(stepsize=0.003)
    cost_tape_assisted, param_tape_assisted = qoptimize.optimize_steps(classical_circuit, assisted_start, iterations=n_iter, opt=opt)

    initial_params = np.random.random(NUM_CPARAMS) * np.pi
    opt = qml.AdamOptimizer(stepsize=0.003)
    cost_tape, param_tape = qoptimize.optimize_steps(classical_circuit, initial_params, iterations=n_iter, opt=opt)

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))



    y0 = moving_average(cost_tape, 20)
    y1 = moving_average(cost_tape_assisted, 20)
    axes[0].plot(range(len(y0)), y0, c='r')
    axes[0].plot(range(len(y1)), y1, c='g')


    axes[1].scatter(param_tape[:,0], param_tape[:,1], c='r', s=5)
    axes[1].scatter(param_tape_assisted[:,0], param_tape_assisted[:,1], c='g', s=5)

    plt.show()

    # qcircuit, _ = rc.make_amplitude_amplification_circuit(num_qparams=NUM_QPARAMS, K=1)
    # qcircuit([1, 2, 3])
    # def get_qcircuit(params):
    #     return circuit(params)
    #
    # # circuit = get_circuit
    # qcircuit = qml.QNode(get_qcircuit, dev)
    # qcircuit([])

    # with open('quantum_circuit_log.txt', 'w') as f:
    #     print("Quantum")
    #     print(qcircuit.draw(), file=f)
    # amplified_probs = ampl_prob_func(cparams=[])

    # probabilities = np.zeros_like(param_0_vals)
    # for idx in np.ndindex(param_0_vals.shape):
    #     probabilities[idx] = amplified_probs[idx]
    #
    #
    # # Evaluate the cost landscape for the random circuit
    # score_landscape = np.ones_like(param_0_vals) * 1e-3
    # for idx in np.ndindex(param_0_vals.shape):
    #     score_landscape[idx] = rc.target_prob([param_0_vals[idx], param_1_vals[idx]])


# Perform optimization on the circuit with quantum parameters