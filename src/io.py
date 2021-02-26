"""io.py - some file mangement for consistent data handling."""

def get_classical_cost_tape_fname(index):
    return "experiments/classical_costs_{}.npy".format(index)

def get_classical_param_tape_fname(index):
    return "experiments/classical_params_{}.npy".format(index)

def get_quantum_cost_tape_fname(index=0):
    return "experiments/quantum_exp_{}.npy".format(index)

def get_quantum_param_tape_fname(index=0):
    return "experiments/quantum_params_{}.npy".format(index)

def get_api_key():
    with open("floq_api.txt", 'r') as fh:
        api = fh.readlines()
    return api[0]