from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

# Qiskit imports
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.providers.aer.noise import NoiseModel
from qiskit.utils import QuantumInstance
from qiskit_aer import AerSimulator

# Qiskit Algorithms imports
from qiskit_algorithms.optimizers import SPSA, ADAM
from qiskit_algorithms.gradients import (
    QFI,  # Quantum Fisher Information
    ReverseUQCGradient,  # Gradient computation method
    SPSAEstimatorGradient,  # SPSA-specific gradient estimation
)
from qiskit_algorithms import AlgorithmJob, EstimatorQAOA

import logging
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor

@dataclass
class OptimizationConfig:
    max_iterations: int = 50
    shots: int = 1024
    p_layers: int = 2
    param_range: Tuple[float, float] = (-np.pi, np.pi)
    threads: int = 4
    save_history: bool = True
    output_dir: Path = Path("optimization_results")

class QAOAGradientOptimizer:
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
""""
    def create_hamiltonian(self) -> SparsePauliOp:
        #Create problem Hamiltonian with improved coefficients
        # !Burası düzenlenecek!
        coefficients = [-1.0, 1.0, -0.5]
        pauli_strings = ["ZZ", "X", "Z"]
        return SparsePauliOp.from_list(list(zip(pauli_strings, coefficients)))
"""
    def create_gradient_methods(self):
        """Create gradient computation methods"""
        # Quantum Fisher Information gradient
        qfi_gradient = QFI()
        
        # Reverse gradient method
        reverse_gradient = ReverseUQCGradient()
        
        # SPSA-specific gradient estimation
        spsa_gradient = SPSAEstimatorGradient(
            num_parameters=self.config.p_layers,
            perturbation=0.1
        )
        
        return {
            "qfi": qfi_gradient,
            "reverse": reverse_gradient,
            "spsa": spsa_gradient
        }

    def optimize(self) -> Dict[str, Any]:
        """Run optimization with gradient-based methods"""
        self.hamiltonian = self.create_hamiltonian()
        self.qaoa_ansatz = QAOAAnsatz(self.hamiltonian, reps=self.config.p_layers)
        
        # Create gradient methods
        gradient_methods = self.create_gradient_methods()
        
        # Create QAOA Estimator
        estimator_qaoa = EstimatorQAOA(
            ansatz=self.qaoa_ansatz,
            estimator=self.quantum_instance
        )

        optimization_results = {}
        
        with ThreadPoolExecutor(max_workers=self.config.threads) as executor:
            # Prepare optimizer configurations
            spsa_optimizer = SPSA(
                maxiter=self.config.max_iterations,
                gradient_method=gradient_methods['spsa']
            )
            
            adam_optimizer = ADAM(
                maxiter=self.config.max_iterations,
                gradient_method=gradient_methods['reverse']
            )
            
            # Initial parameter generation
            initial_params = np.random.uniform(
                self.config.param_range[0],
                self.config.param_range[1],
                len(self.qaoa_ansatz.parameters)
            )

            # Run optimizations
            futures = {
                executor.submit(self._run_optimizer, spsa_optimizer, estimator_qaoa, initial_params, "spsa"): "spsa",
                executor.submit(self._run_optimizer, adam_optimizer, estimator_qaoa, initial_params, "adam"): "adam"
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

    def _run_optimizer(self, optimizer, estimator, initial_params, optimizer_type):
        """Run individual optimizer with gradient methods"""
        try:
            result = optimizer.optimize(
                num_vars=len(initial_params),
                objective_function=estimator.objective_function,
                initial_point=initial_params
            )
            
            return {
                "optimal_params": result[0],
                "optimal_energy": result[1],
                "optimization_history": self.optimization_history[optimizer_type]
            }
        except Exception as e:
            self.logger.error(f"Optimization failed for {optimizer_type}: {e}")
            raise

    def _save_results(self, results: Dict[str, Any]):
        """Save optimization results to file"""
        if self.config.save_history:
            self.config.output_dir.mkdir(exist_ok=True)
            output_file = self.config.output_dir / "optimization_results.json"
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=lambda x: x.tolist())

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
    optimizer = QAOAGradientOptimizer(config)
    
    try:
        results = optimizer.optimize()
        logging.info("Optimization completed successfully")
        return results
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        raise

if __name__ == "__main__":
    main()