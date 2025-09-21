# implementar heuristicas
import numpy as np
import heapq
from evaluar import load_graph, load_workers, INF
import time
import sys
import re

rng = np.random.default_rng(123456)

def dijkstra_single_source(N, adj, source):
    """
    Dijsktra común desde un único source.
    """
    dist = [INF] * N
    pq = []
    dist[source] = 0.0
    heapq.heappush(pq, (0.0, source))
    
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

def calcular_cubiertos(distancias, radios):
    """
    distancias: (W x N) distancia worker->nodo
    radios: (W,) radio por worker
    retorna: (W x N) tal que la pos [i, j]==1 si distancias[i,j] < radios[i] (worker i es cubierto por nodo j)
    """
    radios = np.asarray(radios, float).reshape(-1, 1)
    return np.isfinite(distancias) & (distancias < radios)  # bool (W x N)

def generate_minimal_route(N, adj, i, j):
    """
    Genera el camino mínimo desde el nodo i hasta el nodo j usando Dijkstra.
    """
    dist = [INF] * N
    prev = [None] * N
    pq = []
    dist[i] = 0.0
    heapq.heappush(pq, (0.0, i))
    
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        if u == j:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    
    if dist[j] == INF:
        return []
    
    path = []
    cur = j
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path                           

def calculate_route_cost(route, edge_dict):
    """Calcula el costo total de una ruta dada."""
    total_cost = 0.0
    for k in range(len(route) - 1):
        i, j = route[k], route[k + 1]
        if (i, j) in edge_dict:
            total_cost += edge_dict[(i, j)]
        else:
            total_cost += INF
    return total_cost

def select_nodes(binario, umbral):
    """
    Selecciona nodos que cubren a más de umbral workers para que formen parte de la ruta.
    """
    W, N = binario.shape
    col_counts = binario.sum(axis=0)
    chosen = set(np.flatnonzero(col_counts > umbral))

    covered = binario[:, list(chosen)].any(axis=1) if chosen else np.zeros(W, bool)
    uncovered_idx = np.flatnonzero(~covered)

    while uncovered_idx.size:
        # puntuación: cuántos uncovered cubre cada nodo
        scores = binario[uncovered_idx, :].sum(axis=0)
        scores[list(chosen)] = -1  # excluir ya elegidos
        valid = np.flatnonzero(scores > 0)
        if valid.size == 0:
            # ningún nodo adicional cubre uncovered -> salir
            break

        # 3) tomar aleatoriamente entre los top-k por score (k=10 o menos)
        k = min(10, valid.size)
        # tomar índices de los k mejores sin ordenar completamente (eficiente)
        top_idx_in_valid = np.argpartition(scores[valid], -k)[-k:]
        top_candidates = valid[top_idx_in_valid]

        # elegir uno al azar entre los top-k
        j = int(rng.choice(top_candidates))

        # 4) actualizar conjuntos/estados
        chosen.add(j)
        covered |= binario[:, j]
        uncovered_idx = np.flatnonzero(~covered)

    return np.array(sorted(chosen), dtype=int)

# --- 3) ordena nodos por nearest-neighbor desde 0; inserta caminos si hace falta ---
def _append_path(route, path):
    """Concatena un camino 'path' a la ruta, sin repetir el primer nodo de 'path'."""
    path = np.asarray(path, dtype=int)
    if route[-1] != path[0]:
        raise ValueError(f"El camino a concatenar no empieza en el último nodo de la ruta: {route[-1]} != {path[0]}")
    return np.append(route, path[1:])  # agrega desde el segundo nodo

def coser_ruta(N, adj, nodes):
    nodes = [int(x) for x in nodes]
    if not nodes:
        return np.array([0, 0], dtype=int)

    # 1) elegir 'first' como el nodo más cercano a 0 entre los alcanzables
    d0 = dijkstra_single_source(N, adj, 0)
    reachable = [n for n in nodes if np.isfinite(d0[n])]
    if reachable:
        first = min(reachable, key=lambda x: d0[x])
    else:
        # si ninguno es alcanzable desde 0, no existe ruta factible que empiece en 0
        # podés lanzar error o elegir igual (fallará generate_minimal_route si no hay camino)
        raise ValueError("Ninguno de los nodos seleccionados es alcanzable desde 0; no hay ruta factible.")

    remaining = [n for n in nodes if n != first]

    # 2) ruta inicial: el CAMINO MÍNIMO 0 -> first
    route = np.array([0], dtype=int)
    path_0_first = generate_minimal_route(N, adj, 0, first)  # debe incluir [0, ..., first]
    route = _append_path(route, path_0_first)

    # 3) greedy nearest-neighbor cosiendo SIEMPRE el camino mínimo entre consecutivos
    prev = first
    while remaining:
        dprev = dijkstra_single_source(N, adj, prev)
        nxt = min(remaining, key=lambda x: dprev[x])
        if not np.isfinite(dprev[nxt]):
            raise ValueError(f"No hay camino desde {prev} hasta {nxt}; grafo desconectado.")

        path_prev_nxt = generate_minimal_route(N, adj, prev, nxt)  # [prev, ..., nxt]
        route = _append_path(route, path_prev_nxt)
        prev = nxt
        remaining.remove(nxt)

    # 4) cerrar: coser CAMINO MÍNIMO last -> 0
    path_last_0 = generate_minimal_route(N, adj, route[-1], 0)  # [last, ..., 0]
    route = _append_path(route, path_last_0)

    return route.astype(int)

# --- función principal que integra todo ---
def construir_camino(N, adj, cubiertos):
    """
    N, adj: grafo (adj = lista de adyacencia como devuelve load_graph)
    distancias: (W x N) distancia worker->nodo
    radios: (W,) radio por worker
    umbral: entero (estricto: 'más de umbral')
    """
    # Elegimos umbral random entre 3 y la máxima cantidad de workers cubiertos por un nodo
    umbral = np.random.randint(3, np.max(cubiertos.sum(axis=0)))
    seleccionados = select_nodes(cubiertos, umbral)
    route = coser_ruta(N, adj, seleccionados)
    
    return route, seleccionados

def heuristica(N, adj, cubiertos, dicc_adj, max_attempts=100):
    i = 0
    best_route = None
    best_cost =  INF
    best_selected = None
    while i < max_attempts:
        ruta, selected = construir_camino(N, adj, cubiertos)
        costo = calculate_route_cost(ruta, dicc_adj)
        if costo < best_cost:
            best_cost = costo
            best_route = ruta
            best_selected = selected
        i += 1
    return best_route, best_selected

if __name__ == "__main__":
    
    # Leer input y parsear número de instancia
    if len(sys.argv) != 3:
        print("Ejecutar de la siguiente manera: python main.py <input_graph_path> <input_instanciaX_path>")
        sys.exit(1)
    arg1, arg2 = sys.argv[1], sys.argv[2]
    if not re.match(r'.*\d+\.csv$', arg2):
        print("El segundo argumento debe ser una instancia del tipo 'instanciaX.csv'")
        sys.exit(1)
    else:
        idx = int(re.search(r'instancia(\d+)\.csv$', arg2).group(1))

    # Medir tiempo de ejecución
    start_time = time.time()

    # Cargar datos
    N, edges = load_graph(rf"instancias\{arg1}")
    workers = load_workers(rf"instancias\{arg2}")
    radios = np.array([w[1] for w in workers], dtype=float).reshape(-1, 1)
    distancias = np.vstack([dijkstra_single_source(N, edges, s) for s, _ in workers])
    cubiertos = calcular_cubiertos(distancias, radios)
    ruta, select_nodes = heuristica(N, edges, cubiertos, {(u, v): w for u in range(N) for v, w in edges[u]})

    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    # Escribir solución
    with open(fr'instancias\solucion{idx}.txt', 'w') as f:
        f.write(' '.join(map(str, ruta)))