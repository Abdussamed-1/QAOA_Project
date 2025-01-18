import rustworkx as rx
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from scipy.optimize import minimize
from qiskit_ibm_runtime import SamplerV2 as Sampler
from typing import Sequence


# Ağırlık matrisi
weights_matrix = np.array([
    [0.3, 0.8, 1.6],  # Donor 1 -> Recipients
    [0.4, 1.2, 2.0]   # Donor 2 -> Recipients
])

# Q matrisi 
Q = np.array([
    [-3,  0,  2,  2,  2],
    [ 0, -3,  2,  2,  2],
    [ 0,  0, -2,  0,  0],
    [ 0,  0,  0, -2,  0],
    [ 0,  0,  0,  0, -2]
])

# Atama listesi: x = [1, 1, 0, 0, 0] gruplama
x = np.array([1, 1, 0, 0, 0])  # Donorlar: 1,1 / Alıcılar: 0,0,0

# x^T Q x hesaplama
xt_q_x = np.dot(x.T, np.dot(Q, x))

#print(f"x^T Q x = {xt_q_x}")

# Yeni bir boş graph oluştur
graph = rx.PyGraph()

# Düğümleri ekle
# Donörler (0 ve 1 indeksleri)
donor_indices = [graph.add_node(f"Donor {i+1}") for i in range(weights_matrix.shape[0])]

# Alıcılar (2, 3, 4 indeksleri)
recipient_indices = [graph.add_node(f"Recipient {i+1}") for i in range(weights_matrix.shape[1])]

# Kenarları ekle - ağırlıkları matristen al
weights = {}  # Ağırlıkları saklamak için
for i, donor in enumerate(donor_indices):
    for j, recipient in enumerate(recipient_indices):
        weight = weights_matrix[i, j]  # Matristen ağırlığı al
        graph.add_edge(donor, recipient, weight)
        weights[(donor, recipient)] = weight  # Ağırlığı sakla

# Maksimum kesme ağırlığını hesapla
max_cut_weight = 0
for edge in graph.edge_list():
    u, v = edge[0], edge[1]  # Kenarın iki düğümü
    if x[u] != x[v]:  # Eğer farklı gruptalarsa
        max_cut_weight += weights[(u, v)]

print(f"Maximum cut ağırlığı: {max_cut_weight}")

# Görselleştirme için pozisyonları ayarla
pos = {}

# Donör pozisyonları (sol taraf)
donor_y = np.linspace(0, 1, len(donor_indices))
for idx, donor in enumerate(donor_indices):
    pos[donor] = np.array([0, donor_y[idx]])

# Alıcı pozisyonları (sağ taraf)
recipient_y = np.linspace(0, 1, len(recipient_indices))
for idx, recipient in enumerate(recipient_indices):
    pos[recipient] = np.array([1, recipient_y[idx]])

# Matplotlib figure oluştur
plt.figure(figsize=(5, 6))

# Kenarları çiz ve ağırlıkları ekrana yaz
for edge in graph.edge_list():
    start_pos = pos[edge[0]]
    end_pos = pos[edge[1]]
    color = 'red' if x[edge[0]] != x[edge[1]] else 'gray'  # Kesim kenarlarını vurgula
    plt.plot([start_pos[0], end_pos[0]], 
             [start_pos[1], end_pos[1]], 
             color=color)
    # Ağırlık yazdır
    mid_x = (start_pos[0] + end_pos[0]) / 2
    mid_y = (start_pos[1] + end_pos[1]) / 2
    plt.text(mid_x, mid_y, f"{weights[(edge[0], edge[1])]}",
             color='blue', fontsize=10, ha='center', va='center')

# Düğümleri çiz
for node in graph.node_indices():
    x_coord, y_coord = pos[node]
    color = 'lightblue' if node in donor_indices else 'lightgreen'
    plt.scatter(x_coord, y_coord, c=color, s=500, zorder=2)
    plt.annotate(f"{'Donor' if node in donor_indices else 'Recipient'} {node+1 if node in donor_indices else node-1}", 
                (x_coord, y_coord), 
                xytext=(5, 5), 
                textcoords='offset points',
                ha='center',
                va='center')

# Grafik ayarları
plt.title(f"Bipartite Graph with Maximum Cut\nx^T Q x = {xt_q_x}")
plt.axis('off')
plt.tight_layout()

# Grafı göster
plt.show()

# list 
# QAOA ansatz
# PaulFeature Map
#


def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
    """Convert the graph to Pauli list.

    This function does the inverse of `build_max_cut_graph`
    """
    pauli_list = []
    for edge in list(graph.edge_list()):
        paulis = ["I"] * len(graph)
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"

        weight = graph.get_edge_data(edge[0], edge[1])

        pauli_list.append(("".join(paulis)[::1], weight))

    return pauli_list


max_cut_paulis = build_max_cut_paulis(graph)

cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
#print("Cost Function Hamiltonian:", cost_hamiltonian)



circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2)
circuit.measure_all()

circuit.draw('mpl')

params = circuit.parameters # beta ve gama değerlerini tutmak için bu değerler başlangıç değerleri olacak


QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="bb788c2ba7acfd609700447f1dddb87f08e4ba1c8a195cddf116f536d101eef13320d6f21df40c8ed61b753701e0b661d8c9a777e08a8f2427a0d036b891d76c",  # Replace with your actual token
    overwrite=True,
    set_as_default=True
)

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


# QiskitRuntimeService.save_account(channel="ibm_quantum", token="<MY_IBM_QUANTUM_TOKEN>", overwrite=True, set_as_default=True)
service = QiskitRuntimeService(channel='ibm_quantum')
backend = service.least_busy(min_num_qubits=127)
#print(backend)

# Create pass manager for transpilation
pm = generate_preset_pass_manager(optimization_level=3,
                                    backend=backend)

candidate_circuit = pm.run(circuit)
candidate_circuit.draw('mpl', fold=False, idle_wires=False)

def cost_func_estimator(params, ansatz, hamiltonian, estimator):

    # transform the observable defined on virtual qubits to
    # an observable defined on all physical qubits
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])

    results = job.result()[0]
    cost = results.data.evs

    objective_func_vals.append(cost)
    return cost

objective_func_vals = []  # Global variable
with Session(backend=backend) as session:
    # If using qiskit-ibm-runtime<0.24.0, change `mode=` to `session=
    
    estimator = Estimator(mode=session)
    estimator.options.default_shots = 1000

    # Set simple error suppression/mitigation options
    
    estimator.options.dynamical_decoupling.enable = True
    estimator.options.dynamical_decoupling.sequence_type = "XY4"
    estimator.options.twirling.enable_gates = True
    estimator.options.twirling.num_randomizations = "auto"

    result = minimize(
        cost_func_estimator,
        init_params,
        args=(candidate_circuit, cost_hamiltonian, estimator),
        method="COBYLA",
        tol=1e-2,
    )
    print(result)

plt.figure(figsize=(12, 6))  # cost - iteration graphics
plt.plot(objective_func_vals)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()


# If using qiskit-ibm-runtime<0.24.0, change `mode=` to `backend=`
sampler = Sampler(mode=backend)
sampler.options.default_shots = 10000

# Set simple error suppression/mitigation options
sampler.options.dynamical_decoupling.enable = True
sampler.options.dynamical_decoupling.sequence_type = "XY4"
sampler.options.twirling.enable_gates = True
sampler.options.twirling.num_randomizations = "auto"

pub= (optimized_circuit,)
job = sampler.run([pub], shots=int(1e4))
counts_int = job.result()[0].data.meas.get_int_counts()
counts_bin = job.result()[0].data.meas.get_counts()
shots = sum(counts_int.values())
final_distribution_int = {key: val/shots for key, val in counts_int.items()}
final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}
#print(final_distribution_int)

# auxiliary functions to sample most likely bitstring
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]

keys = list(final_distribution_int.keys())
values = list(final_distribution_int.values())
most_likely = keys[np.argmax(np.abs(values))]
most_likely_bitstring = to_bitstring(most_likely, len(graph))
most_likely_bitstring.reverse()

#print("Result bitstring:", most_likely_bitstring)

matplotlib.rcParams.update({"font.size": 10})

final_bits = final_distribution_bin
values = np.abs(list(final_bits.values()))
top_4_values = sorted(values, reverse=True)[:4]
positions = []
for value in top_4_values:
    positions.append(np.where(values == value)[0])
fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(1, 1, 1)
plt.xticks(rotation=45)
plt.title("Result Distribution")
plt.xlabel("Bitstrings (reversed)")
plt.ylabel("Probability")
ax.bar(list(final_bits.keys()), list(final_bits.values()), color="tab:grey")
for p in positions:
    ax.get_children()[int(p)].set_color("tab:purple")
plt.show()

def evaluate_sample(x: Sequence[int], graph: rx.PyGraph) -> float:
    assert len(x) == len(list(graph.nodes())), "The length of x must coincide with the number of nodes in the graph."
    return sum(x[u] * (1 - x[v]) + x[v] * (1 - x[u]) for u, v in list(graph.edge_list()))


cut_value= evaluate_sample(most_likely_bitstring, graph)
# print('The value of the cut is:', cut_value)