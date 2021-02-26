"""qarameterize.py: Some utility functions for demonstrating qarameterization."""

import pennylane as qml
from pennylane import numpy as np


class RandomCircuit:
    gate_set = [qml.RX, qml.RY, qml.RZ]
    ctrl_gate_set = [qml.CRX, qml.CRY, qml.CRZ]

    def __init__(self,
                 num_qubits,
                 precision_qparams=5,
                 max_qparams=2 * np.pi,
                 include_max=False,
                 random_gateidx_sequence=None,
                 seed=None):
        self.num_qubits = self.num_params = num_qubits
        self.precision_qparams = precision_qparams
        self.max_qparams = max_qparams

        self.num_bins = 2**precision_qparams
        if include_max:
            self.num_bins -= 1

        if random_gateidx_sequence is None:
            if seed is not None:
                np.random.seed(seed)
            self.random_gateidx_sequence = np.random.choice(3, self.num_params)
        else:
            self.random_gateidx_sequence = random_gateidx_sequence

        self.devices = [
            qml.device('default.qubit',
                       wires=num_qubits + i * precision_qparams)
            for i in range(self.num_params + 1)
        ]

        self.__init_target_prob__()

    @qml.template
    def rand_circuit(self, cparams, num_qparams):
        # Numbers of classical and "quantum" parameters
        num_cparams = self.num_params - num_qparams

        # Wires:
        #   First `num_qubits` wires form the main register.
        #   After that, every `precision_qparams` wires coresponds to one qparam.

        num_wires = self.num_qubits + num_qparams * self.precision_qparams

        for i in range(self.num_qubits):
            qml.RY(np.pi / 4, wires=i)

        for i in range(num_cparams):
            self.gate_set[self.random_gateidx_sequence[i]](cparams[i], wires=i)

        for i in range(num_cparams, self.num_params):
            ctrl_gate = self.ctrl_gate_set[self.random_gateidx_sequence[i]]
            for j in range(self.precision_qparams):
                ctrl_wire = self.num_qubits + (
                    i - num_cparams) * self.precision_qparams + j

                #j = 0 is the least significant
                ctrl_angle = self.max_qparams * 2**(self.precision_qparams -
                                                    j - 1) / self.num_bins

                ctrl_gate(ctrl_angle, wires=[ctrl_wire, i])

        for i in range(self.num_qubits - 1):
            qml.CZ(wires=[i, i + 1])

    def __init_target_prob__(self):
        @qml.qnode(self.devices[0])
        def func(cparams):
            self.rand_circuit(cparams, num_qparams=0)

            H = np.zeros((2**self.num_qubits, 2**self.num_qubits))
            H[0, 0] = 1
            return qml.expval(qml.Hermitian(H, wires=range(self.num_qubits)))

        self.target_prob = func

    # Quantum oracle to evaluate the state of the control registers
    @qml.template
    def oracle(self, cparams, num_qparams):
        diag = np.ones(2**self.num_qubits)
        diag[0] *= -1

        qml.inv(self.rand_circuit(cparams, num_qparams))
        qml.DiagonalQubitUnitary(diag, wires=range(self.num_qubits))

        self.rand_circuit(cparams, num_qparams)
        qml.DiagonalQubitUnitary(diag, wires=range(self.num_qubits))

    # Non-Boolean Amplitude Amplification circuit
    def make_amplitude_amplification_circuit(self, num_qparams, K):
        assert num_qparams <= self.num_params
        num_wires = self.num_qubits + num_qparams * self.precision_qparams

        @qml.template
        def init_circuit(cparams):
            # Initialize control registers to a uniform superposition state
            for i in range(self.num_qubits, num_wires):
                qml.Hadamard(wires=i)

            # Couple the control registers to main register (initialize main
            # register basedon the state of the control)
            self.rand_circuit(cparams, num_qparams)

        @qml.qnode(self.devices[num_qparams])
        def circuit_aux(cparams):
            diag = np.ones(2**num_wires)
            diag[0] *= -1

            # Initialization
            init_circuit(cparams)

            # Non-Boolean Amplitude Amplification go brrrrr
            for k in range(1, K + 1):
                # Act the oracle during odd iterations, or its inverse during
                # the even iterations
                if k % 2 == 1:
                    self.oracle(cparams, num_qparams)
                else:
                    qml.inv(self.oracle(cparams, num_qparams))

                # Diffusion operator
                qml.inv(init_circuit(cparams))
                qml.DiagonalQubitUnitary(diag, wires=range(num_wires))
                init_circuit(cparams)

            # Measurement of the control registers post amplification
            return qml.probs(wires=range(self.num_qubits, num_wires))

        def amplified_probs(cparams):
            probs = circuit_aux(cparams)
            probs_dict = {}
            for i in range(len(probs)):
                tmp, key = i, []
                for j in range(num_qparams):
                    key.append(tmp % 2**self.precision_qparams)
                    tmp //= 2**self.precision_qparams

                key = tuple(key)
                probs_dict[key] = float(probs[i])
            return probs_dict

        return circuit_aux, amplified_probs
        np.random.choice()


class LayerwiseRandomCircuit:
    gate_set = [qml.RX, qml.RY, qml.RZ]
    ctrl_gate_set = [qml.CRX, qml.CRY, qml.CRZ]

    def __init__(self,
                 num_qubits,
                 num_params,
                 precision_qparams=5,
                 max_qparams=2 * np.pi,
                 include_max=False,
                 random_gateidx_sequence=None,
                 seed=None):
        self.num_qubits = num_qubits
        self.num_params = num_params
        self.precision_qparams = precision_qparams
        self.max_qparams = max_qparams

        self.num_bins = 2**precision_qparams
        if include_max:
            self.num_bins -= 1

        if random_gateidx_sequence is None:
            if seed is not None:
                np.random.seed(seed)
            self.random_gateidx_sequence = np.random.choice(3, self.num_params)
        else:
            self.random_gateidx_sequence = random_gateidx_sequence

        np.random.seed(seed)
        # Just a big enough static list of gate choices
        self.singleq_gate_sequence = np.random.choice(3,
                                                      size=10 * num_params *
                                                      num_qubits)

        self.devices = [
            qml.device('default.qubit',
                       wires=num_qubits + i * precision_qparams)
            for i in range(self.num_params + 1)
        ]

        self.__init_target_prob__()

    @qml.template
    def rand_circuit(self, cparams, num_qparams):
        # Numbers of classical and "quantum" parameters
        gate_idx = 0
        num_cparams = self.num_params - num_qparams

        # Wires:
        #   First `num_qubits` wires form the main register.
        #   After that, every `precision_qparams` wires coresponds to one qparam.

        num_wires = self.num_qubits + num_qparams * self.precision_qparams

        # Iterate over layers
        for k in range(self.num_params):
            # Algorithm description: the number of layers in this circuit is
            # equal to num_params, with each layer receiving a single parameter.
            # The first layers will be filled with classical parameters, after
            # which the remainder of layers receive quantum parameters.

            # sqrt(H) mixing layer
            for i in range(self.num_qubits):
                qml.RY(np.pi / 4, wires=i)

            # Fill out classical parameter first
            if k < num_cparams:
                # First wire gets the param
                self.gate_set[self.random_gateidx_sequence[k]](cparams[k],
                                                               wires=0)
            # Fill out quantum parameters
            else:
                ctrl_gate = self.ctrl_gate_set[self.random_gateidx_sequence[k]]
                # Add controls to every wire in the ancilla register according
                # to the bit of theta that we're capturing.
                for j in range(self.precision_qparams):
                    ctrl_wire = self.num_qubits + (
                        k - num_cparams) * self.precision_qparams + j

                    #j = 0 is the least significant
                    ctrl_angle = self.max_qparams * 2**(
                        self.precision_qparams - j - 1) / self.num_bins

                    ctrl_gate(ctrl_angle, wires=[ctrl_wire, 0])
            # Pad the remainder of each parameterized layer with random paulis
            for i in range(1, self.num_qubits):
                rand_angle = np.random.random() * np.pi
                rand_gate = self.gate_set[self.singleq_gate_sequence[gate_idx]]
                gate_idx += 1
                rand_gate(rand_angle, wires=i)

            # Interleaved entangling gates
            for i in range(0, self.num_qubits - 1, 2):
                qml.CZ(wires=[i, i + 1])
            for i in range(1, self.num_qubits - 1, 2):
                qml.CZ(wires=[i, i + 1])

    def __init_target_prob__(self):
        @qml.qnode(self.devices[0])
        def func(cparams):
            self.rand_circuit(cparams, num_qparams=0)

            H = np.zeros((2**self.num_qubits, 2**self.num_qubits))
            H[0, 0] = 1
            return qml.expval(qml.Hermitian(H, wires=range(self.num_qubits)))

        self.target_prob = func

    # Quantum oracle to evaluate the state of the control registers
    @qml.template
    def oracle(self, cparams, num_qparams):
        diag = np.ones(2**self.num_qubits)
        diag[0] *= -1

        qml.inv(self.rand_circuit(cparams, num_qparams))
        qml.DiagonalQubitUnitary(diag, wires=range(self.num_qubits))

        self.rand_circuit(cparams, num_qparams)
        qml.DiagonalQubitUnitary(diag, wires=range(self.num_qubits))

    # Non-Boolean Amplitude Amplification circuit
    def make_amplitude_amplification_circuit(self, num_qparams, K):
        assert num_qparams <= self.num_params
        num_wires = self.num_qubits + num_qparams * self.precision_qparams

        @qml.template
        def init_circuit(cparams):
            # Initialize control registers to a uniform superposition state
            for i in range(self.num_qubits, num_wires):
                qml.Hadamard(wires=i)

            # Couple the control registers to main register (initialize main
            # register basedon the state of the control)
            self.rand_circuit(cparams, num_qparams)

        @qml.qnode(self.devices[num_qparams])
        def circuit_aux(cparams):
            diag = np.ones(2**num_wires)
            diag[0] *= -1

            # Initialization
            init_circuit(cparams)

            # Non-Boolean Amplitude Amplification go brrrrr
            for k in range(1, K + 1):
                # Act the oracle during odd iterations, or its inverse during
                # the even iterations
                if k % 2 == 1:
                    self.oracle(cparams, num_qparams)
                else:
                    qml.inv(self.oracle(cparams, num_qparams))

                # Diffusion operator
                qml.inv(init_circuit(cparams))
                qml.DiagonalQubitUnitary(diag, wires=range(num_wires))
                init_circuit(cparams)

            # Measurement of the control registers post amplification
            return qml.probs(wires=range(self.num_qubits, num_wires))

        def amplified_probs(cparams):
            probs = circuit_aux(cparams)
            probs_dict = {}
            for i in range(len(probs)):
                tmp, key = i, []
                for j in range(num_qparams):
                    key.append(tmp % 2**self.precision_qparams)
                    tmp //= 2**self.precision_qparams

                key = tuple(key)
                probs_dict[key] = float(probs[i])
            return probs_dict

        return circuit_aux, amplified_probs


def compute_K(generator):
    """Compute the K hyperparameter controlling # iterations of amplification.

    Args:
        generator: An initialized circuit generator (one of the ones above)
    """
    np.random.seed(0)
    N = 50
    target_prob_list = [
        generator.target_prob([
            np.random.uniform(low=0, high=2 * np.pi)
            for j in range(generator.num_qubits)
        ]) for _ in range(N)
    ]
    cosphi_list = [2 * p - 1 for p in target_prob_list]
    cosPhi, cosPhi_error = np.mean(
        cosphi_list), np.std(cosphi_list) / np.sqrt(N)
    Phi = np.arccos(cosPhi)

    K = int(np.pi / (2 * (np.pi - Phi)))
    return K
