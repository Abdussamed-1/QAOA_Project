import rustworkx as rx
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from scipy.optimize import minimize
from qiskit_ibm_runtime import QiskitRuntimeService, Session, Estimator, Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


###############################################
# 1. KLASİK PROBLEM TANIMI ve ÖRNEK x^T Q x
###############################################

# A) Ağırlık matrisi (Donor x Recipient)
weights_matrix = np.array([
    [1.0, 1.0, 1.0],  # Donor 1 -> Recipients
    [1.0, 1.0, 1.0]   # Donor 2 -> Recipients
])

# B) Q matrisi (klasik cost ifadesinde)
Q = np.array([
    [-3,  0,  2,  2,  2],
    [ 0, -3,  2,  2,  2],
    [ 0,  0, -2,  0,  0],
    [ 0,  0,  0, -2,  0],
    [ 0,  0,  0,  0, -2]
])

# Örnek bitstring (Donorlar=1,1; Recipients=0,0,0)
x = np.array([1, 1, 0, 0, 0])
xt_q_x = x.T @ Q @ x
print(f"x^T Q x = {xt_q_x}")


##################################################
# 2. GRAF OLUŞTURMA ve GÖRSELLEŞTİRME (Bipartite)
##################################################

# PyGraph oluştur
graph = rx.PyGraph()

# Donörler (2 adet)
donor_indices = [graph.add_node(f"Donor {i+1}") for i in range(weights_matrix.shape[0])]
# Alıcılar (3 adet)
recipient_indices = [graph.add_node(f"Recipient {j+1}") for j in range(weights_matrix.shape[1])]

# Kenarları ekle
weights = {}
for i, donor in enumerate(donor_indices):
    for j, recipient in enumerate(recipient_indices):
        w = weights_matrix[i, j]
        graph.add_edge(donor, recipient, w)
        weights[(donor, recipient)] = w

# x'e göre Maximum cut ağırlığı (sadece görsel)
max_cut_weight = 0
for (u, v) in graph.edge_list():
    if x[u] != x[v]:
        max_cut_weight += weights[(u, v)]
print(f"Maximum cut ağırlığı: {max_cut_weight}")

# Basit bir yerleşim düzeni
pos = {}
donor_y = np.linspace(0, 1, len(donor_indices))
for idx, donor in enumerate(donor_indices):
    pos[donor] = np.array([0, donor_y[idx]])
recipient_y = np.linspace(0, 1, len(recipient_indices))
for idx, recipient in enumerate(recipient_indices):
    pos[recipient] = np.array([1, recipient_y[idx]])

plt.figure(figsize=(5, 6))

for (u, v) in graph.edge_list():
    start_pos = pos[u]
    end_pos = pos[v]
    color = 'red' if x[u] != x[v] else 'gray'
    plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color=color)
    mid_x = (start_pos[0] + end_pos[0]) / 2
    mid_y = (start_pos[1] + end_pos[1]) / 2
    plt.text(mid_x, mid_y, f"{weights[(u, v)]}",
             color='blue', fontsize=10, ha='center', va='center')

for node in graph.node_indices():
    xx, yy = pos[node]
    color = 'lightblue' if node in donor_indices else 'lightgreen'
    plt.scatter(xx, yy, c=color, s=500, zorder=2)
    label = f"{'Donor' if node in donor_indices else 'Recipient'} {node+1 if node in donor_indices else node-1}"
    plt.annotate(label, (xx, yy), xytext=(5, 5),
                 textcoords='offset points', ha='center', va='center')

plt.title(f"Bipartite Graph - Example Cut\nx^T Q x = {xt_q_x}")
plt.axis('off')
plt.tight_layout()
plt.show()


########################################
# 3. KENARLARA GÖRE COST HAMILTONYENİ
########################################
# Kullanıcı "ZIZII", "ZIIZI", "ZIIIZ", "IZZII", "IZIZI", "IZIIZ" terimleri
# ve coeffs=[1,1,1,1,1,1] şeklinde istemiştir.
# Aşağıdaki fonksiyon, tam olarak 2 Donor + 3 Recipient kenarlarını
# (0,2), (0,3), (0,4), (1,2), (1,3), (1,4) ekler ve SparsePauliOp döndürür.

def build_bipartite_z_terms(donor_indices, recipient_indices, total_qubits=5):
    """
    Her kenar için Z_i * Z_j terimi oluşturur, katsayı = +1.
    2 donor + 3 recipient'tan oluşan bipartite düzeni varsayıyoruz:
       donor_indices = [0, 1]
       recipient_indices = [2, 3, 4]
    Toplam qubit sayısı = 5 (0..4)
    """
    paulis = []
    for d in donor_indices:
        for r in recipient_indices:
            # Örn: donor=0, recipient=2 => 'ZIZII'
            # Qubit 0 ve 2'de 'Z', geri kalan 'I'
            z_list = ['I'] * total_qubits
            z_list[d] = 'Z'
            z_list[r] = 'Z'
            p_str = "".join(z_list)
            # katsayı = +1
            paulis.append((p_str, 1.0 + 0.j))

    return SparsePauliOp.from_list(paulis)

# Oluştur ve yazdır
cost_hamiltonian = build_bipartite_z_terms(
    donor_indices=[0,1],
    recipient_indices=[2,3,4],
    total_qubits=5
)
print("Cost Function Hamiltonian:\n", cost_hamiltonian)


########################################
# 4. QAOA Devresini Oluşturma
########################################
from qiskit.circuit.library import QAOAAnsatz

# reps=1 veya reps=2 -> katman sayısı
circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=2, initial_state=None)
circuit.measure_all()

print("QAOA circuit parameters:", circuit.parameters)
circuit.draw("mpl")


########################################
# 5. IBM Runtime Bağlantısı & Transpile
########################################
# (Token bilginizi buraya girin)
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="7cfd0137a84c245cf996bc1e5b71b4738300ff9290fc86e75eb6be677de0953b92981326b9e63e482c432a9c007e6a71d57317f2f3ed613540bd265a040c926b",
    overwrite=True, 
    set_as_default=True
)

service = QiskitRuntimeService(channel='ibm_quantum')
backend = service.least_busy(min_num_qubits=7)  # 5 qubit yeterli olduğu için 7 qubit minimum diyelim
print("Kullanılacak backend:", backend)

pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
candidate_circuit = pm.run(circuit)
candidate_circuit.draw("mpl", fold=False, idle_wires=False)


########################################
# 6. QAOA Parametre Optimizasyonu
########################################
from scipy.optimize import minimize

beta = np.pi
gamma = np.pi / 2
init_params = [beta, gamma, beta, gamma]  # reps=2 => 4 parametre

objective_func_vals = []  # iterasyon sırasında cost değerlerini saklayalım

def cost_func_estimator(params, ansatz, hamiltonian, estimator):
    # QAOA ansatz layout'unu alalım (Sabit kalsın, genelde 1e1)
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])
    results = job.result()[0]
    cost = results.data.evs
    objective_func_vals.append(cost)
    return cost

with Session(backend=backend) as session:
    estimator = Estimator(mode=session)
    estimator.options.default_shots = 1024

    result = minimize(
        cost_func_estimator,
        init_params,
        args=(candidate_circuit, cost_hamiltonian, estimator),
        method="COBYLA",
        tol=1e-3
    )

    print("Optimizasyon sonucu:")
    print(result)

plt.figure(figsize=(10,5))
plt.plot(objective_func_vals, marker='o')
plt.title("QAOA Cost vs Iteration")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

optimized_circuit = candidate_circuit.assign_parameters(result.x)
optimized_circuit.draw("mpl", fold=False, idle_wires=False)


########################################
# 7. Sonuç Ölçümü ve Bitstring Analizi
########################################
with Session(backend=backend) as session:
    sampler = Sampler(session=session)
    sampler.options.default_shots = 1024

    job = sampler.run([optimized_circuit], shots=10_000)
    meas_result = job.result()[0].data.meas
    counts_int = meas_result.get_int_counts()
    counts_bin = meas_result.get_counts()

    shots = sum(counts_int.values())
    final_distribution_int = {key: val/shots for key, val in counts_int.items()}
    final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}

print("Final measurement distribution (integer keys):")
print(final_distribution_int)

# En olası bitstring'i bul
def to_bitstring(integer, num_bits):
    return list(map(int, np.binary_repr(integer, width=num_bits)))

keys = list(final_distribution_int.keys())
vals = list(final_distribution_int.values())
most_likely_key = keys[np.argmax(vals)]
most_likely_bits = to_bitstring(most_likely_key, 5)
most_likely_bits.reverse()  # Qiskit bit ordering
print("Most likely bitstring (reversed):", most_likely_bits)

# Çubuk grafiği
matplotlib.rcParams.update({"font.size": 10})
plt.figure(figsize=(10, 5))
plt.title("Measurement distribution (bitstrings reversed)")
plt.bar(list(final_distribution_bin.keys()), list(final_distribution_bin.values()), color="tab:gray")
plt.xticks(rotation=45)
plt.show()

# Basit kesim değeri değerlendirme (graf'a göre 0..4 arası qubit correspond to nodes)
def evaluate_sample(x_bits: Sequence[int], graph: rx.PyGraph) -> float:
    return sum(x_bits[u] != x_bits[v] for (u, v) in graph.edge_list())

cut_value = evaluate_sample(most_likely_bits, graph)
print("Found cut value:", cut_value)
