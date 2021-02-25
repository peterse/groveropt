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

        # One parameter per layer
        n_layers = self.num_params

        # Wires:
        #   First `num_qubits` wires form the main register.
        #   After that, every `precision_qparams` wires coresponds to one qparam.

        num_wires = self.num_qubits + num_qparams * self.precision_qparams

        for k in range(n_layers):
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
                self.gate_set[self.random_gateidx_sequence[k]](cparams[k], wires=0)
                for i in range(1, self.num_qubits):
                    rand_angle = np.random.random() * np.pi
                    rand_gate = np.random.choice(self.gate_set)
                    rand_gate(rand_angle, wires=0)
            # Then fill out quantum parameters
            else:
                ctrl_gate = self.ctrl_gate_set[self.random_gateidx_sequence[k]]
                for j in range(self.precision_qparams):
                    ctrl_wire = self.num_qubits + (
                        k - num_cparams) * self.precision_qparams + j

                    #j = 0 is the least significant
                    ctrl_angle = self.max_qparams * 2**(self.precision_qparams -
                                                        j - 1) / self.num_bins

                    ctrl_gate(ctrl_angle, wires=[ctrl_wire, 0])

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
