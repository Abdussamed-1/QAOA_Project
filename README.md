# QAOA Project - Quantum Optimization for Donor-Recipient Matching

This project implements the Quantum Approximate Optimization Algorithm (QAOA) to solve a bipartite graph maximum cut problem, specifically applied to donor-recipient matching scenarios. The implementation uses Qiskit and IBM Quantum services to run quantum circuits.

## Project Overview

The project focuses on solving an optimization problem where we need to match donors with recipients optimally. The problem is formulated as a bipartite graph where:
- Nodes are divided into two sets: donors and recipients
- Edges represent potential matches with associated weights
- The goal is to find the optimal matching that maximizes the total weight of the cut

## Key Features

- Implementation of QAOA algorithm for quantum optimization
- Support for both local simulation and IBM Quantum backend execution
- Visualization of bipartite graphs and maximum cuts
- Cost function optimization using classical-quantum hybrid approach
- Integration with IBM Quantum services for real quantum hardware execution

## Project Structure

- `Main.py`: Main implementation file containing the core QAOA algorithm and graph construction
- `simulator.py`: Contains simulation-specific implementations and local testing capabilities
- `Donör-Recipient/`: Directory containing donor-recipient specific implementations
- `main/`: Additional implementation files and utilities

## Requirements

- Python 3.x
- Qiskit
- IBM Quantum Account
- NumPy
- Matplotlib
- Rustworkx
- SciPy

## Setup

1. Install required packages:
```bash
pip install qiskit qiskit-ibm-runtime numpy matplotlib rustworkx scipy
```

2. Set up IBM Quantum credentials:
   - Create an account at [IBM Quantum](https://quantum-computing.ibm.com/)
   - Get your API token
   - Configure the credentials in the code

## Usage

1. Run the main implementation:
```bash
python Main.py
```

2. For local simulation:
```bash
python simulator.py
```

## Implementation Details

The project implements:
- Bipartite graph construction and visualization
- Cost Hamiltonian formulation
- QAOA circuit construction
- Parameter optimization
- Quantum circuit execution (both local and IBM Quantum)

## License

This project is open source and available under the MIT License.

