from evaluar import load_graph, load_workers
from heuristica_num1 import resolver
import networkx as nx
import time

start_time = time.time()

N, adj = load_graph("instancias\grafo.csv")
G = nx.DiGraph()
for u in range(N):
    for v, c in adj[u]:
        G.add_edge(u, v, weight=c)


for i in range(10):
    trabajadores = load_workers(f"instancias\instancia{i+1}.csv")
    # num iteraciones, alpha, segs, instancias
    ruta = resolver(200, 0.2, 30, i+1, G, N, adj, trabajadores)
    print(ruta)

end_time = time.time()
print(f"Execution time: {end_time - start_time:.2f} seconds")