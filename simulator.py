import rustworkx as rx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from typing import Sequence
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from scipy.optimize import minimize

# IBM Quantum runtime için:
from qiskit_ibm_runtime import QiskitRuntimeService

# AerSimulator ve FakeKyoto
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import GenericBackendV2

# Qiskit temel fonksiyonlar (transpile vb.)
from qiskit import transpile


# ---------------------------------------------------------------------
# BACKEND SEÇİMİ
# ---------------------------------------------------------------------
# 1) IBM Quantum'a bağlanmak istiyorsanız (örn. "ibm_kyoto"):
#service = QiskitRuntimeService()
#backend_real_kyoto = service.backend()

# 2) Lokal (gürültüsüz) simülatör
backend_aer = AerSimulator()

# 3) FakeKyoto (ibm_kyoto benzeri bir mock/gürültü modeli):
backend_fake = GenericBackendV2(num_qubits=128)
sim_fake = AerSimulator.from_backend(backend_fake)

# ---------------------------------------------------------------------
# Burada hangi backend'i kullanacağınızı seçiyorsunuz:
# (Yorum satırını kaldırarak dilediğiniz backend'i "backend" olarak ayarlayabilirsiniz)
# ---------------------------------------------------------------------
# backend = backend_real_kyoto   # GERÇEK IBM Quantum cihazına gönderir
# backend = backend_aer          # Lokal, gürültüsüz simülatör
backend = backend_aer       # FakeKyoto gürültü modeline sahip simülatör


# ---------------------------------------------------------------------
# MaxCut / QAOA Problemi
# ---------------------------------------------------------------------

# Ağırlık matrisi
weights_matrix = np.array([
    [1.0, 1.0, 1.0],  # Donor 1 -> Recipients
    [1.0, 1.0, 1.0]   # Donor 2 -> Recipients
])

# Q matrisi 
Q = np.array([
    [-3,  0,  1,  1,  1],
    [ 0, -3,  1,  1,  1],
    [ 1,  1, -2,  0,  0],
    [ 1,  1,  0, -2,  0],
    [ 1,  1,  0,  0, -2]
])

# Örnek atama listesi: x = [1, 1, 0, 0, 0]
x = np.array([1, 1, 0, 0, 0])  # Donorlar: 1,1 / Alıcılar: 0,0,0

# x^T Q x hesaplama
xt_q_x = np.dot(x.T, np.dot(Q, x))
print(f"x^T Q x = {xt_q_x}")

# Rustworkx grafı kuralım
graph = rx.PyGraph()
donor_indices = [graph.add_node(f"Donor {i+1}") for i in range(weights_matrix.shape[0])]
recipient_indices = [graph.add_node(f"Recipient {i+1}") for i in range(weights_matrix.shape[1])]

weights = {}
for i, donor in enumerate(donor_indices):
    for j, recipient in enumerate(recipient_indices):
        weight = weights_matrix[i, j]
        graph.add_edge(donor, recipient, weight)
        weights[(donor, recipient)] = weight

# x'e göre (kontrol amaçlı) maximum cut ağırlığı
max_cut_weight = 0
for edge in graph.edge_list():
    u, v = edge[0], edge[1]
    if x[u] != x[v]:
        max_cut_weight += weights[(u, v)]
print(f"Maximum cut ağırlığı: {max_cut_weight}")

# ---------------------------------------------------------------------
# Graf görselleştirme (opsiyonel)
# ---------------------------------------------------------------------
pos = {}
donor_y = np.linspace(0, 1, len(donor_indices))
for idx, donor in enumerate(donor_indices):
    pos[donor] = np.array([0, donor_y[idx]])

recipient_y = np.linspace(0, 1, len(recipient_indices))
for idx, recipient in enumerate(recipient_indices):
    pos[recipient] = np.array([1, recipient_y[idx]])

plt.figure(figsize=(5, 6))
for edge in graph.edge_list():
    start_pos = pos[edge[0]]
    end_pos = pos[edge[1]]
    color = 'red' if x[edge[0]] != x[edge[1]] else 'gray'
    plt.plot([start_pos[0], end_pos[0]],
             [start_pos[1], end_pos[1]],
             color=color)
    mid_x = (start_pos[0] + end_pos[0]) / 2
    mid_y = (start_pos[1] + end_pos[1]) / 2
    plt.text(mid_x, mid_y, f"{weights[(edge[0], edge[1])]}",
             color='blue', fontsize=10, ha='center', va='center')

for node in graph.node_indices():
    x_coord, y_coord = pos[node]
    color = 'lightblue' if node in donor_indices else 'lightgreen'
    plt.scatter(x_coord, y_coord, c=color, s=500, zorder=2)
    plt.annotate(
        f"{'Donor' if node in donor_indices else 'Recipient'} "
        f"{node+1 if node in donor_indices else node-1}",
        (x_coord, y_coord),
        xytext=(5, 5),
        textcoords='offset points',
        ha='center',
        va='center'
    )

plt.title(f"Bipartite Graph with Maximum Cut\nx^T Q x = {xt_q_x}")
plt.axis('off')
plt.tight_layout()
plt.show()

# Cost Hamiltonian
def build_max_cut_paulis(g: rx.PyGraph) -> list[tuple[str, float]]:
    pauli_list = []
    for edge in list(g.edge_list()):
        paulis = ["I"] * len(g)
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"
        weight = g.get_edge_data(edge[0], edge[1])
        pauli_list.append(("".join(paulis), weight))
    return pauli_list

max_cut_paulis = build_max_cut_paulis(graph)
cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)
print("Cost Function Hamiltonian:", cost_hamiltonian)

# QAOA devresi oluştur
circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2, initial_state=None)
circuit.measure_all()
circuit.draw('mpl')
print("Parametreler:", circuit.parameters)

# Devreyi backend'e göre transpile edelim
transpiled_circuit = transpile(circuit, backend=backend, optimization_level=3)

# Başlangıç parametreleri
beta =  np.pi /2 
gamma =  np.pi 
init_params = [beta, gamma, beta, gamma]

# Yardımcı fonksiyonlar
def evaluate_sample(x_bits: Sequence[int], g: rx.PyGraph) -> float:
    """Her bitstring'e karşılık max-cut değerini hesaplar."""
    return sum(x_bits[u] * (1 - x_bits[v]) + x_bits[v] * (1 - x_bits[u])
               for u, v in list(g.edge_list()))

def measure_cost_from_counts(counts, g: rx.PyGraph):
    """Counts sözlüğünden ortalama (expected) kesme değerini bulur."""
    total_shots = sum(counts.values())
    exp_val = 0.0
    for bitstring, cnt in counts.items():
        # Devrede qubit 0 "sağ" tarafta olabileceği için bitstring'i reverse edebiliriz:
        x_array = np.array(list(map(int, bitstring[::-1])))
        cost_per_shot = evaluate_sample(x_array, g)
        exp_val += cost_per_shot * cnt
    return exp_val / total_shots

# Optimizasyon fonksiyonu
objective_func_vals = []

def cost_func_estimator(params, base_circuit, g):
    param_dict = dict(zip(base_circuit.parameters, params))
    bound_circuit = base_circuit.assign_parameters(param_dict)
    
    job = backend.run(bound_circuit, shots=1024)
    counts = job.result().get_counts()

    # Gerçek (pozitif) kesim değerini hesapla
    real_cost = measure_cost_from_counts(counts, g)

    # Grafikte göstermek için bu değeri kaydediyoruz:
    objective_func_vals.append(real_cost)

    # Optimizasyon fonksiyonu ise negatifini döndürüyor
    # Çünkü 'minimize' fonksiyonunu kullanıyoruz.
    return real_cost


# COBYLA ile optimize et
result = minimize(
    cost_func_estimator,
    init_params,
    args=(transpiled_circuit, graph),
    method="COBYLA",
    tol=1e-4
)
print("\nOptimizasyon Sonucu:", result)
print("Optimize Edilmiş Parametreler:", result.x)

# Optimizasyon süreci
plt.figure(figsize=(8, 4))
plt.plot(objective_func_vals, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Cost (MaxCut)")
plt.title("COBYLA Optimizasyon Süreci (Gerçek Cost Değerleri)")
plt.show()

# Optimize devreyle final ölçüm
optimized_circuit = transpiled_circuit.assign_parameters(
    dict(zip(transpiled_circuit.parameters, result.x))
)

job = backend.run(optimized_circuit, shots=10_000)
final_counts = job.result().get_counts()
shots = sum(final_counts.values())
final_distribution = {k: v/shots for k, v in final_counts.items()}

# En olası bitstring
keys = list(final_distribution.keys())
vals = list(final_distribution.values())
most_likely = keys[np.argmax(vals)]
most_likely_bitstring = list(map(int, most_likely[::-1]))

print("\nEn olası (most likely) bitstring:", most_likely)
cut_value = evaluate_sample(most_likely_bitstring, graph)
print("Bu bitstring için kesme (cut) değeri:", cut_value)

# Sonuç dağılımı
matplotlib.rcParams.update({"font.size": 10})
sorted_dist = dict(sorted(final_distribution.items(), key=lambda x: x[1], reverse=True))
plt.figure(figsize=(10, 4))
plt.bar(sorted_dist.keys(), sorted_dist.values(), color="tab:grey")
plt.title("Bitstring Dağılımı (Seçilen Backend Sonucu)")
plt.xlabel("Bitstring (sağ -> qubit 0)")
plt.ylabel("Olasılık")
plt.xticks(rotation=45)
plt.show()
