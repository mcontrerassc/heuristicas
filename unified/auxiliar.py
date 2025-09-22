import numpy as np
import heapq
import time

# COSTO TOTAL DE LA RUTA 
def costear_ruta(ruta, distancias):
    # simil al evaluar.py
    costo_total = 0
    nodo_actual = 0
    for nodo in ruta:
        cost = distancias[nodo_actual, nodo]
        if cost == float('inf'):
            return float('inf')
        costo_total += cost
        nodo_actual = nodo
    costo_total += distancias[nodo_actual, 0]
    return costo_total

# MEJORA 
def mejora_2opt(ruta, distancias, time_limit=None):
    if len(ruta) < 2:
        return ruta

    mejorada = True
    mejor_ruta = ruta.copy()
    mejor_costo = costear_ruta(ruta, distancias)

    while mejorada:
        if time_limit is not None and time.time() > time_limit:
            break
        mejorada = False
        for i in range(len(ruta)):
            for j in range(i + 2, len(ruta)):
                # intercambiamos segmentos de la ruta
                nueva_ruta = ruta[:i+1] + list(reversed(ruta[i+1:j+1]))+ ruta[j+1:]
                nuevo_costo = costear_ruta(nueva_ruta, distancias)
                if nuevo_costo < mejor_costo:
                    mejor_ruta = nueva_ruta.copy()
                    mejor_costo = nuevo_costo
                    mejorada = True
        ruta = mejor_ruta.copy()

    return mejor_ruta, mejor_costo

# COSER CAMINO ENTRE DOS NODOS DE UNA RUTA 
# f. auxiliar
def generate_minimal_route(N, adj, i, j):
    """
    Genera el camino mínimo desde el nodo i hasta el nodo j usando Dijkstra.
    """
    dist = [float('inf')] * N
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
    
    if dist[j] == float('inf'):
        return []
    
    path = []
    cur = j
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path                           

def _append_path(route, path):
    """Concatena un camino 'path' a la ruta, sin repetir el primer nodo de 'path'."""
    path = np.asarray(path, dtype=int)
    if route[-1] != path[0]:
        raise ValueError(f"El camino a concatenar no empieza en el último nodo de la ruta: {route[-1]} != {path[0]}")
    return np.append(route, path[1:])  # agrega desde el segundo nodo

# f. principal 
def coser_camino(N, adj, principio, fin, ruta): 
    path_prev_nxt = generate_minimal_route(N, adj, principio, fin)  # [principio, ..., fin]
    ruta = _append_path(ruta, path_prev_nxt)
    return ruta

# CONSTRUIR CAMINO FINAL

def construir_camino(N, adj, nodes):
    # una vez tenemos los puntos pickup, calculamos la ruta completa
    ruta_final = [0]  # empieza en la empresa
    actual = 0

    for siguiente_pickup in nodes:
        ruta_final = coser_camino(N, adj, actual, siguiente_pickup, ruta_final)
        actual = siguiente_pickup

    # segmento final para volver a la empresa
    ruta_final = coser_camino(N, adj, actual, 0, ruta_final)

    return ruta_final