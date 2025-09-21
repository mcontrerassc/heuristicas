from evaluar import load_graph, load_workers
from heuristica_num1 import resolver
import networkx as nx

N, adj = load_graph("grafo.csv")
G = nx.DiGraph()
for u in range(N):
    for v, c in adj[u]:
        G.add_edge(u, v, weight=c)

trabajadores = load_workers("instancia1.csv")
for i in range(10):
    trabajadores = load_workers(f"instancia{i+1}.csv")
    # num iteraciones, alpha, segs, instancias
    ruta = resolver(200, 0.2, 30, i+1, G, N, adj, trabajadores)
    print(ruta)