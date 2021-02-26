# Qarameterized circuits: Quantum parameters for QML

**Evan Peters**, University of Waterloo, Institute for Quantum Computing  
**Prasanth Shyamsundar**, Fermi National Accelerator Laboratory, Fermilab Quantum Institute

This project is a submission to the **QHack Open Hackathon 2021**.  
Team Name: **PhaseliciousDeinonyqus**

For a presentation of this project please go to our [Presentation Page](https://peterse.github.io/groveropt).

### Project Description: 

Typically, variational quantum circuits are parameterized by **classical** parameters and the circuit is evaluated by minimizing an observable-based cost function using **classical** optimization techniques.

> What if we parameterize quantum circuits using **quantum parameters**?
Can we train such circuits in a manifestly quantum manner?

Enter **Qarameterized Circuits** (Quantum-parameterized Circuits). In this project we

1. Construct variational circuits parameterized by control quantum registers, whose computational basis states correspond to different values for the circuit parameters.
2. Construct a quantum oracle to *coherently* evaluate the state of the control registers, based on a chosen cost function.
3. Train the circuit using a modified version of Grover's algorithm, which preferentially amplifies the good states of the control registers.

This project builds on the findings of Prasanth Shaymsundar, _"Non-Boolean Quantum Amplitude Amplification and Quantum Mean Estimation"_ (2018), [arXiv:2102.04975 [quant-ph]](https://arxiv.org/abs/2102.04975).