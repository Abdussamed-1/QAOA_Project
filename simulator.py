import rustworkx as rx
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
from scipy.optimize import minimize

# Ağırlık matrisi
weights_matrix = np.array([
    [1.0, 1.0, 1.0],  # Donor 1 -> Recipients
    [1.0, 1.0, 1.0]   # Donor 2 -> Recipients
])
# Q matrisi 
Q = np.array([
    [-3,  0,  2,  2,  2],
    [ 0, -3,  2,  2,  2],
    [ 0,  0, -2,  0,  0],
    [ 0,  0,  0, -2,  0],
    [ 0,  0,  0,  0, -2]
])

# Graph oluşturma
graph = rx.PyGraph()

# Düğümleri ekle
num_qubits = 5
donor_indices = [graph.add_node(f"Donor {i+1}") for i in range(weights_matrix.shape[0])]
recipient_indices = [graph.add_node(f"Recipient {i+1}") for i in range(weights_matrix.shape[1])]

# Kenarları ekle
weights = {}
for i, donor in enumerate(donor_indices):
    for j, recipient in enumerate(recipient_indices):
        weight = weights_matrix[i, j]
        graph.add_edge(donor, recipient, weight)
        weights[(donor, recipient)] = weight

def create_qaoa_circuit(params, num_qubits):
    """
    Create QAOA circuit manually
    params: [teta0, teta1]
    """
    # Başlangıç durumu: tüm qubitleri Hadamard geçişi
    circuit = QuantumCircuit(num_qubits)
    circuit.h(range(num_qubits))
    
    # QAOA katmanları
    teta0 , teta1 = params
    
    # Cost Hamiltonian katmanı (1. iterasyon)
    for edge in graph.edge_list():
        u, v = edge[0], edge[1]
        circuit.rzz(2 * teta0 * weights[(u, v)], u, v)

    # Mixing Hamiltonian (1. iterasyon)
    circuit.rx(2 * teta1, range(num_qubits))

    # Cost Hamiltonian katmanı (2. iterasyon)
    for edge in graph.edge_list():
        u, v = edge[0], edge[1]
        circuit.rzz(2 * teta0 * weights[(u, v)], u, v)

    # Mixing Hamiltonian (2. iterasyon)
    circuit.rx(2 * teta1, range(num_qubits)) 
    
    # Ölçüm
    circuit.measure_all()
    
    return circuit

# Simulator
simulator = AerSimulator()

teta0 = np.pi 
teta1 = np.pi / 2 

initial_params = [teta0,teta1]


# Maliyet fonksiyonu
objective_func_vals = []
def cost_function(params):
    # QAOA devresini oluştur
    qaoa_circuit = create_qaoa_circuit(params, num_qubits)
    
    # Simülasyon
    job = simulator.run(qaoa_circuit, shots=100)
    result = job.result()
    counts = result.get_counts()
    
    # Maliyet hesaplama
    total_cost = 0
    total_shots = sum(counts.values())
    
    for bitstring, count in counts.items():
        # Bitstringi tersine çevir (MSB sağda olacak şekilde)
        x = np.array([int(b) for b in bitstring[::-1]])
        
        # x^T Q x hesapla
        xt_q_x = np.dot(x.T, np.dot(Q, x))
        
        # Maliyete katkıda bulun
        total_cost += xt_q_x * (count / total_shots)
    
    objective_func_vals.append(total_cost)
    return total_cost

# Optimizasyon
result = minimize(
    cost_function,
    initial_params,
    method="COBYLA",
    tol=1e-3,
)

print("Optimize edilmiş parametreler:", result.x)
print("Minimum maliyet:", result.fun)

# En iyi parametrelerle son simülasyon
final_circuit = create_qaoa_circuit(result.x, num_qubits)
job = simulator.run(final_circuit, shots=100)
result = job.result()
counts = result.get_counts()

# Histogramın boş olmaması için:
if counts:
    plot_histogram(counts)
else:
    print("Simülasyon sonuçları boş! Grafiği görüntülemek mümkün değil.")

# Sonuçları görselleştir
plt.figure(figsize=(15, 6))
plot_histogram(counts)
plt.title("QAOA Max Cut Sonuç Dağılımı")
plt.show()

# Optimizasyon sürecini görselleştir
plt.figure(figsize=(12, 6))
plt.plot(objective_func_vals)
plt.title("Optimizasyon Sürecinde Maliyet Fonksiyonu Değerleri")
plt.xlabel("İterasyon")
plt.ylabel("Maliyet")
plt.show()

# En olası bitstring'i bul
max_bitstring = max(counts, key=counts.get)
print("En Olası Bitstring:", max_bitstring)
print("Frekans:", counts[max_bitstring] / 1000),