from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.providers.aer.noise import NoiseModel
from qiskit.utils import QuantumInstance
from qiskit_aer import AerSimulator
from qiskit.algorithms.optimizers import SPSA, ADAM
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import json

@dataclass
class OptimizationConfig:
    max_iterations: int = 50
    shots: int = 1024
    p_layers: int = 2
    param_range: Tuple[float, float] = (-np.pi, np.pi)
    threads: int = 4
    save_history: bool = True
    output_dir: Path = Path("optimization_results")

class QAOAOptimizer:
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.setup_logging()
        self.initialize_quantum_backend()
        self.optimization_history = {
            "spsa": [],
            "adam": []
        }
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def initialize_quantum_backend(self):
        try:
            noise_model = NoiseModel()
            # Add realistic noise parameters here
            self.backend = AerSimulator(noise_model=noise_model)
            self.quantum_instance = QuantumInstance(
                backend=self.backend,
                shots=self.config.shots,
                optimization_level=3
            )
            self.logger.info("Quantum backend initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize quantum backend: {e}")
            raise

    def create_hamiltonian(self) -> SparsePauliOp:
        """Create problem Hamiltonian with improved coefficients"""
        coefficients = [-1.0, 1.0, -0.5]  # More complex problem
        pauli_strings = ["ZZ", "X", "Z"]
        return SparsePauliOp.from_list(list(zip(pauli_strings, coefficients)))

    def compute_energy(self, params: List[float]) -> float:
        """Compute energy with error handling and validation"""
        try:
            if len(params) != len(self.qaoa_ansatz.parameters):
                raise ValueError("Invalid parameter dimension")
                
            param_dict = dict(zip(self.qaoa_ansatz.parameters, params))
            bound_circuit = self.qaoa_ansatz.bind_parameters(param_dict)
            
            statevector = Statevector.from_instruction(bound_circuit)
            expectation_value = np.real(
                np.vdot(statevector.data, 
                        self.hamiltonian.to_matrix() @ statevector.data)
            )
            return expectation_value
        except Exception as e:
            self.logger.error(f"Energy computation failed: {e}")
            raise

    def optimize(self) -> Dict[str, Any]:
        """Run optimization with multiple optimizers in parallel"""
        self.hamiltonian = self.create_hamiltonian()
        self.qaoa_ansatz = QAOAAnsatz(self.hamiltonian, reps=self.config.p_layers)

        optimization_results = {}
        
        with ThreadPoolExecutor(max_workers=self.config.threads) as executor:
            futures = {
                executor.submit(self._run_optimizer, "spsa"): "spsa",
                executor.submit(self._run_optimizer, "adam"): "adam"
            }
            
            for future in futures:
                optimizer_name = futures[future]
                try:
                    result = future.result()
                    optimization_results[optimizer_name] = result
                except Exception as e:
                    self.logger.error(f"{optimizer_name} optimization failed: {e}")

        self._save_results(optimization_results)
        self._plot_results()
        
        return optimization_results

    def _run_optimizer(self, optimizer_type: str) -> Dict[str, Any]:
        """Run individual optimizer with appropriate configuration"""
        if optimizer_type == "spsa":
            optimizer = SPSA(maxiter=self.config.max_iterations)
        else:
            optimizer = ADAM(maxiter=self.config.max_iterations)

        initial_params = np.random.uniform(
            self.config.param_range[0],
            self.config.param_range[1],
            len(self.qaoa_ansatz.parameters)
        )

        result = optimizer.optimize(
            num_vars=len(initial_params),
            objective_function=self.compute_energy,
            initial_point=initial_params
        )
        
        return {
            "optimal_params": result[0],
            "optimal_energy": result[1],
            "optimization_history": self.optimization_history[optimizer_type]
        }

    def _save_results(self, results: Dict[str, Any]):
        """Save optimization results to file"""
        if self.config.save_history:
            self.config.output_dir.mkdir(exist_ok=True)
            output_file = self.config.output_dir / "optimization_results.json"
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    def _plot_results(self):
        """Create and save visualization of optimization results"""
        plt.figure(figsize=(12, 6))
        for optimizer in self.optimization_history:
            plt.plot(
                self.optimization_history[optimizer],
                label=optimizer.upper()
            )
        
        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        plt.title("Energy Optimization Comparison")
        plt.legend()
        plt.grid(True)
        
        if self.config.save_history:
            plt.savefig(self.config.output_dir / "optimization_plot.png")
        plt.show()

def main():
    """Main execution function"""
    config = OptimizationConfig()
    optimizer = QAOAOptimizer(config)
    
    try:
        results = optimizer.optimize()
        logging.info("Optimization completed successfully")
        return results
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        raise

if __name__ == "__main__":
    main()