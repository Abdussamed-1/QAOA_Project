from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import rustworkx as rx
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_ibm_runtime import QiskitRuntimeService, Session, EstimatorV2, SamplerV2
from scipy.optimize import minimize
import logging

@dataclass
class MaxCutConfig:
    num_donors: int
    num_recipients: int
    qaoa_reps: int = 2
    shots: int = 1000
    optimization_level: int = 3
    token: str = ""
    
class QuantumMaxCut:
    def __init__(self, config: MaxCutConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.graph = None
        self.weights_matrix = None
        self.cost_hamiltonian = None
        self.circuit = None
        self.objective_func_vals = []
        
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    def initialize_weights(self, weights: Optional[np.ndarray] = None) -> None:
        """Initialize or validate weight matrix"""
        if weights is None:
            # Generate random weights if none provided
            self.weights_matrix = np.random.uniform(
                0.1, 2.0, 
                (self.config.num_donors, self.config.num_recipients)
            )
        else:
            if weights.shape != (self.config.num_donors, self.config.num_recipients):
                raise ValueError(f"Weight matrix shape {weights.shape} does not match configuration")
            self.weights_matrix = weights
            
    def build_graph(self) -> None:
        """Construct bipartite graph from weights"""
        try:
            self.graph = rx.PyGraph()
            
            # Add nodes with validation
            donor_indices = [
                self.graph.add_node({
                    'type': 'donor',
                    'id': i,
                    'label': f"Donor {i+1}"
                }) for i in range(self.config.num_donors)
            ]
            
            recipient_indices = [
                self.graph.add_node({
                    'type': 'recipient',
                    'id': i + self.config.num_donors,
                    'label': f"Recipient {i+1}"
                }) for i in range(self.config.num_recipients)
            ]
            
            # Add edges with weights
            self.weights = {}
            for i, donor in enumerate(donor_indices):
                for j, recipient in enumerate(recipient_indices):
                    weight = self.weights_matrix[i, j]
                    self.graph.add_edge(donor, recipient, weight)
                    self.weights[(donor, recipient)] = weight
                    
            self.logger.info(f"Graph built successfully with {len(donor_indices)} donors and {len(recipient_indices)} recipients")
        except Exception as e:
            self.logger.error(f"Error building graph: {e}")
            raise
            
    def build_qaoa_circuit(self) -> None:
        """Build QAOA circuit from graph"""
        try:
            pauli_list = self._build_max_cut_paulis()
            self.cost_hamiltonian = SparsePauliOp.from_list(pauli_list)
            self.circuit = QAOAAnsatz(
                cost_operator=self.cost_hamiltonian,
                reps=self.config.qaoa_reps
            )
            self.circuit.measure_all()
            self.logger.info(f"QAOA circuit built with {self.config.qaoa_reps} repetitions")
        except Exception as e:
            self.logger.error(f"Error building QAOA circuit: {e}")
            raise
            
    def _build_max_cut_paulis(self) -> List[Tuple[str, float]]:
        """Convert graph to Pauli operators"""
        pauli_list = []
        for edge in self.graph.edge_list():
            paulis = ["I"] * len(self.graph)
            paulis[edge[0]], paulis[edge[1]] = "Z", "Z"
            weight = self.graph.get_edge_data(edge[0], edge[1])
        
            pauli_list.append(("".join(paulis)[::1], weight))
        return pauli_list
        
    def optimize_circuit(self, backend_name: str = None) -> Dict:
        """Run quantum optimization"""
        try:
            service = QiskitRuntimeService(channel='ibm_quantum', token=self.config.token)
            backend = service.least_busy(min_num_qubits=5) if backend_name is None else service.backend(backend_name)
            
            with Session(backend=backend) as session:
                estimator = EstimatorV2(mode=session)
                estimator.options.default_shots = self.config.shots
                
                # Configure error mitigation
                estimator.options.dynamical_decoupling.enable = True
                estimator.options.twirling.enable_gates = True
                
                # Optimization
                init_params = np.random.uniform(0, 2*np.pi, len(self.circuit.parameters))
                result = minimize(
                    self._cost_function_estimator,
                    init_params,
                    args=(self.circuit, self.cost_hamiltonian, estimator),
                    method="COBYLA",
                    tol=1e-2
                )
                
                return {
                    'optimal_parameters': result.x,
                    'optimal_value': result.fun,
                    'success': result.success,
                    'message': result.message
                }
                
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            raise
            
    def _cost_function_estimator(self, params, ansatz, hamiltonian, estimator):
        """Estimate cost function value"""
        try:
            isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
            job = estimator.run([(ansatz, isa_hamiltonian, params)])
            result = job.result()[0]
            cost = result.data.evs
            self.objective_func_vals.append(cost)
            return cost
        except Exception as e:
            self.logger.error(f"Cost estimation failed: {e}")
            raise
            
    def plot_results(self) -> None:
        """Visualize optimization results"""
        try:
            # Plot optimization progress
            plt.figure(figsize=(12, 6))
            plt.plot(self.objective_func_vals)
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("QAOA Optimization Progress")
            plt.grid(True)
            plt.show()
            
            # Plot graph structure
            self._plot_graph()
            
        except Exception as e:
            self.logger.error(f"Plotting failed: {e}")
            raise
            
    def _plot_graph(self) -> None:
        """Plot the bipartite graph structure"""
        pos = self._calculate_graph_positions()
        plt.figure(figsize=(8, 6))
        
        # Plot edges
        for edge in self.graph.edge_list():
            start_pos = pos[edge[0]]
            end_pos = pos[edge[1]]
            plt.plot([start_pos[0], end_pos[0]], 
                    [start_pos[1], end_pos[1]], 
                    color='gray', alpha=0.6)
            
        # Plot nodes
        for node in self.graph.node_indices():
            x_coord, y_coord = pos[node]
            node_data = self.graph.get_node_data(node)
            color = 'lightblue' if node_data['type'] == 'donor' else 'lightgreen'
            plt.scatter(x_coord, y_coord, c=color, s=500, zorder=2)
            plt.annotate(node_data['label'], 
                        (x_coord, y_coord),
                        ha='center', va='center')
            
        plt.title("Quantum MaxCut Problem Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def main():
    # Configuration
    config = MaxCutConfig(
        num_donors=2,
        num_recipients=3,
        qaoa_reps=2,
        shots=1000,
        token="your_ibm_token_here"
    )
    
    # Initialize solver
    solver = QuantumMaxCut(config)
    
    # Example weights
    weights = np.array([
        [0.3, 0.8, 1.6],
        [0.4, 1.2, 2.0]
    ])
    
    # Run optimization
    solver.initialize_weights(weights)
    solver.build_graph()
    solver.build_qaoa_circuit()
    result = solver.optimize_circuit()
    
    # Plot results
    solver.plot_results()
    
    return result

if __name__ == "__main__":
    main()