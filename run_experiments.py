import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np

from src import qoptimize, qarameterize, io

NUM_QUBITS = 4
NUM_PARAMS = 4 # Assume either all classical or all quantum

PRECISION = 4
SEED = 3717

local_dev = qml.device('default.qubit', wires=range(NUM_QUBITS), analytic=True)

USE_FLOQ = False
if USE_FLOQ:
    import remote_cirq
    floq_sim = remote_cirq.RemoteSimulator(io.get_api_key())
    grover_devices = [
        # Analytic=False is required for some weird compatibility reason
        qml.device("cirq.simulator", wires=NUM_QUBITS + i * PRECISION, simulator=floq_sim, analytic=False) for i in range(NUM_PARAMS + 1)
    ]
else:
    dev = local_dev
    grover_devices = None
print("Using Floq? {}".format(USE_FLOQ))



def main():

    # Query the quantum parameter landscape, optionally with floq
    generator_q = qarameterize.LayerwiseRandomCircuit(num_qubits=NUM_QUBITS, num_params= NUM_PARAMS, precision_qparams=PRECISION, seed=SEED, devices=grover_devices, floq=USE_FLOQ)
    K_opt = qarameterize.compute_K(generator_q)
    print("Performing {} iterations".format(K_opt))
    ampl_circuit, ampl_prob_func = generator_q.make_amplitude_amplification_circuit(num_qparams=NUM_PARAMS, K=K_opt)
    amplified_probs = ampl_prob_func(cparams=[])
    print("...Finished grover iterations")
    # Sample from the grover-assisted distribution of thetas
    param_vals = np.linspace(0, 2*np.pi, num=2**PRECISION, endpoint=False)
    shape = (2 ** PRECISION, ) * NUM_PARAMS

    # Create a grid of probabilities
    probabilities = np.zeros(shape)
    for idx in np.ndindex(shape):
        probabilities[idx] = amplified_probs[idx]
    p_flat = probabilities.flatten()

    # Sample from the grid
    nsamples = int(5e6)
    idx_samples = np.random.choice(np.arange(len(p_flat)), p=p_flat, size=nsamples)
    idx_samples = [np.unravel_index(x, shape) for x in idx_samples]
    theta_samples = []
    # Generate the lists of parameters associated with every sample
    for idx in idx_samples:
        theta_samples.append([param_vals[i] for i in idx])
    print("...Finished sampling outcome")
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

    classical_circuit = qml.QNode(get_classical_circuit, local_dev)


    # qml.AdamOptimizer(stepsize=0.003)
    def make_optimizer():
        return qml.AdamOptimizer(stepsize=0.003)
        # return qml.RMSPropOptimizer(stepsize=0.001, decay=0.9, eps=1e-08)
    # Optimization
    n_trials = 5
    n_iter = 500

    print("Quantum training:")
    assisted_start = np.asarray(theta_samples).mean(axis=0)
    opt = make_optimizer()
    cost_tape_assisted, param_tape_assisted = qoptimize.optimize_steps(classical_circuit, assisted_start, iterations=n_iter, opt=opt)

    np.save(io.get_quantum_cost_tape_fname(), cost_tape_assisted)
    np.save(io.get_quantum_param_tape_fname(), param_tape_assisted)
    np.random.seed(SEED) # Make sure to keep consistent seed for the random samples
    for k in range(n_trials):
        print("Classical Training, trial {}".format(k))
        random_start = np.random.random(NUM_PARAMS) * np.pi
        opt = make_optimizer()
        cost_tape, param_tape = qoptimize.optimize_steps(classical_circuit, random_start, iterations=n_iter, opt=opt)
        np.save(io.get_classical_cost_tape_fname(k), cost_tape)
        np.save(io.get_classical_param_tape_fname(k), param_tape)

if __name__ == "__main__":
    main()