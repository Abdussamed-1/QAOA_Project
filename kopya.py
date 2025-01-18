from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.utils import QuantumInstance
from qiskit_aer import AerSimulator
from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from hyperopt import fmin, tpe, hp, Trials
import numpy as np
import matplotlib.pyplot as plt

# Kuantum ortamı (gürültü modellemesi için QASM simülatörü)
backend = AerSimulator()
quantum_instance = QuantumInstance(backend=backend, shots=1024)

# Hamiltonyen tanımı (örnek bir problem için Ising modelini ele alalım)
coefficients = [-1, 1]
pauli_strings = ["ZZ", "X"]
hamiltonian = SparsePauliOp.from_list(list(zip(pauli_strings, coefficients)))

# QAOA ansatz tanımı
p = 2  # QAOA'nın derinlik seviyesi
qaoa_ansatz = QAOAAnsatz(hamiltonian, reps=p)

# Optimizasyon sonuçlarını takip için global değişkenler
optimization_history_smac = []
optimization_history_tpe = []

# Enerji fonksiyonu
def compute_energy(params):
    param_dict = dict(zip(qaoa_ansatz.parameters, params))
    bound_circuit = qaoa_ansatz.bind_parameters(param_dict)

    # Beklenti değeri hesaplama
    statevector = Statevector.from_instruction(bound_circuit)
    expectation_value = np.real(np.vdot(statevector.data, hamiltonian.to_matrix() @ statevector.data))

    return expectation_value

# SMAC ile QAOA optimizasyonu
# Konfigürasyon uzayı tanımlama
cs = ConfigurationSpace()
for i in range(len(qaoa_ansatz.parameters)):
    cs.add_hyperparameter(UniformFloatHyperparameter(f"param_{i}", -np.pi, np.pi))

# SMAC hedef fonksiyonu
def smac_objective(config):
    params = [config[f"param_{i}"] for i in range(len(qaoa_ansatz.parameters))]
    energy = compute_energy(params)
    optimization_history_smac.append(energy)
    return energy

# SMAC senaryosu
scenario = Scenario({
    "run_obj": "quality",  # Optimize edilecek hedef
    "runcount-limit": 50,  # Maksimum iterasyon sayısı
    "cs": cs  # Parametre uzayı
})

# SMAC optimizasyonu
smac = SMAC(scenario=scenario, tae_runner=smac_objective)
incumbent = smac.optimize()
optimal_params_smac = [incumbent[f"param_{i}"] for i in range(len(qaoa_ansatz.parameters))]
optimal_energy_smac = compute_energy(optimal_params_smac)

# TPE ile QAOA optimizasyonu
# TPE hedef fonksiyonu
def tpe_objective(params):
    energy = compute_energy(params)
    optimization_history_tpe.append(energy)
    return energy

# Hyperopt uzayı tanımlama
space = [hp.uniform(f"param_{i}", -np.pi, np.pi) for i in range(len(qaoa_ansatz.parameters))]

# TPE optimizasyonu
trials = Trials()
tpe_result = fmin(
    fn=tpe_objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trials
)
optimal_params_tpe = [tpe_result[f"param_{i}"] for i in range(len(qaoa_ansatz.parameters))]
optimal_energy_tpe = compute_energy(optimal_params_tpe)

# Optimizasyon sonuçlarını görselleştirme
plt.plot(optimization_history_smac, label="SMAC")
plt.plot(optimization_history_tpe, label="TPE")
plt.xlabel("İterasyon")
plt.ylabel("Enerji")
plt.title("Enerji Değeri Optimizasyonu")
plt.legend()
plt.show()

# Optimizasyon sonuçlarını yazdırma
print("SMAC Optimal Parametreler:", optimal_params_smac)
print("SMAC Optimal Enerji:", optimal_energy_smac)
print("TPE Optimal Parametreler:", optimal_params_tpe)
print("TPE Optimal Enerji:", optimal_energy_tpe)

# Parametre yüzeyi görselleştirme
if len(qaoa_ansatz.parameters) == 2:  # Yalnızca iki parametre için uygundur
    param_1, param_2 = np.meshgrid(np.linspace(-np.pi, np.pi, 50), np.linspace(-np.pi, np.pi, 50))
    energies = np.array([[compute_energy([p1, p2]) for p2 in param_2[0]] for p1 in param_1[:, 0]])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(param_1, param_2, energies, cmap='viridis')
    plt.title("Parametre Yüzeyi")
    plt.show()