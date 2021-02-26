import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np

import remote_cirq
from src import qoptimize, qarameterize, io

NUM_QUBITS = 4
NUM_PARAMS = 4 # Assume either all classical or all quantum

PRECISION = 5
SEED = 11

USE_FLOQ = True
if USE_FLOQ:
    floq_sim = remote_cirq.RemoteSimulator(io.get_api_key())
    dev = qml.device("cirq.simulator", wires=range(NUM_QUBITS), simulator=floq_sim)
else:
    dev = qml.device('default.qubit', wires=range(NUM_QUBITS), shots=1000, analytic=False)
print("Using Floq? {}".format(USE_FLOQ))



def main():

    # # Force that total number of params equals number of qparams to push out any
    # # classical param inputs
    generator_q = qarameterize.LayerwiseRandomCircuit(num_qubits=NUM_QUBITS, num_params= NUM_PARAMS, precision_qparams=PRECISION, seed=SEED)
    K_opt = qarameterize.compute_K(generator_q)
    ampl_circuit, ampl_prob_func = generator_q.make_amplitude_amplification_circuit(num_qparams=NUM_PARAMS, K=K_opt)
    amplified_probs = ampl_prob_func(cparams=[])

    # Sample from the grover-assisted distribution of thetas
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


    # Compare optimization with and without grover iterations
    rc = qarameterize.LayerwiseRandomCircuit(num_qubits=NUM_QUBITS,
                                             num_params=NUM_PARAMS,
                                             precision_qparams=PRECISION,
                                             seed=SEED)

    def get_classical_circuit(params):
        rc.rand_circuit(params, num_qparams=0)
        H = np.zeros((2 ** NUM_QUBITS, 2 ** NUM_QUBITS))
        H[0, 0] = 1
        wirelist = [i for i in range(NUM_QUBITS)]
        return qml.expval(qml.Hermitian(H, wirelist))

    classical_circuit = qml.QNode(get_classical_circuit, dev)


    # Optimization
    n_trials = 5
    n_iter = 300

    print("Quantum training:")
    assisted_start = np.asarray(theta_samples).mean(axis=0)
    opt = qml.AdamOptimizer(stepsize=0.003)
    cost_tape_assisted, param_tape_assisted = qoptimize.optimize_steps(classical_circuit, assisted_start, iterations=n_iter, opt=opt)

    np.save(io.get_quantum_cost_tape_fname(), cost_tape_assisted)
    np.save(io.get_quantum_param_tape_fname(), param_tape_assisted)
    for k in range(n_trials):
        print("Classical Training, trial {}".format(k))
        random_start = np.random.random(NUM_PARAMS) * np.pi
        opt = qml.AdamOptimizer(stepsize=0.003)
        cost_tape, param_tape = qoptimize.optimize_steps(classical_circuit, random_start, iterations=n_iter, opt=opt)
        np.save(io.get_classical_cost_tape_fname(k), cost_tape)
        np.save(io.get_classical_param_tape_fname(k), param_tape)

if __name__ == "__main__":
    main()