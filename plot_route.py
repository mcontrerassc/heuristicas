import sys
import os
from evaluar import parse_route_txt, load_graph, load_workers
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Ejecutar de la siguiente manera: python plot_route.py <indice de ruta>")
        sys.exit(1)
    idx = sys.argv[1]

    # Leer archivo CSV
    try:
        gpath = os.path.join(f"instancias/grafo.csv")
        ipath = os.path.join(f"instancias/instancia{idx}.csv")
        spath = os.path.join(f"instancias/solucion{idx}.txt")
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        sys.exit(1)

    # Cargar datos
    N, adj = load_graph(gpath)
    workers = load_workers(ipath)
    W = len(workers)
    route = parse_route_txt(spath)

    # Reestructurar datos para networkx
    # Nodos
    workers_for_G = [(v, {"radio": r, "color": "red"}) for v, r in workers]
    rest_of_nodes = set(range(N)) - set(v for v, r in workers)
    rest_of_nodes_for_G = [(v, {"color": "black"}) for v in rest_of_nodes]
    # Aristas
    dicc_aristas = {(u, v): w for u in range(N) for v, w in adj[u]}
    route_edges_for_G = [(route[i], route[i + 1], {"color": "blue", "weight": dicc_aristas[(route[i], route[i + 1])]}) for i in range(len(route) - 1)]
    rest_of_edges_for_G = [(u, v, {"weight": w}) for (u, v), w in dicc_aristas.items() if (u,v) not in [(route[i], route[i + 1]) for i in range(len(route) - 1)]]

    # Armar el grafo
    G = nx.Graph()
    G.add_nodes_from(workers_for_G)
    G.add_nodes_from(rest_of_nodes_for_G)
    G.add_edges_from(route_edges_for_G)
    G.add_edges_from(rest_of_edges_for_G)
    posicion = nx.spring_layout(G)

    # Configurar la visualización
    nodes_order = list(G.nodes())
    node_colors = [G.nodes[n].get("color", "black") for n in nodes_order]

    route_pairs = set(map(frozenset, zip(route[:-1], route[1:])))
    edges_order = list(G.edges())
    edge_colors = ['blue' if frozenset(e) in route_pairs else 'gray' for e in edges_order]
    edge_widths = [2.5  if frozenset(e) in route_pairs else 1.5  for e in edges_order]
    
    # Dibujar el grafo
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, posicion, nodelist=nodes_order,
                        node_color=node_colors, node_size=10, linewidths=0.2)
    nx.draw_networkx_edges(G, posicion, edgelist=edges_order,
                        edge_color=edge_colors, width=edge_widths, alpha=0.9)
    nx.draw_networkx_labels(G, posicion, font_size=7)

    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"imagenes/solucion_{idx}_route.png")