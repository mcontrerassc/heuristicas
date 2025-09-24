import numpy as np
import itertools
import heapq
import time
import random

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
        return None  # no hay camino
    
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
    if path_prev_nxt == None:
        return None  # no hay camino posible
    ruta = _append_path(ruta, path_prev_nxt)
    return ruta

# CONSTRUIR CAMINO FINAL

def construir_camino(N, adj, nodes):
    # una vez tenemos los puntos pickup, calculamos la ruta completa
    ruta_final = [0]  # empieza en la empresa
    actual = 0

    for siguiente_pickup in nodes:
        ruta_final = coser_camino(N, adj, actual, siguiente_pickup, ruta_final)
        if ruta_final is None:
            return None  # no hay camino posible
        actual = siguiente_pickup

    # segmento final para volver a la empresa
    ruta_final = coser_camino(N, adj, actual, 0, ruta_final)
    if ruta_final is None:
        return None  # no hay camino posible

    return ruta_final

# Evaluar exclusión de un trabajador

def evaluate_exclusion_combinations(N, adj, distancias, workers, cubiertos, ruta, mejor_costo, r=5):
    maxP = 0

    for r in range(1, r + 1):
        print(f"Evaluando exclusión de combinaciones de {r} trabajadores...")
        combinaciones = list(itertools.combinations(range(len(workers)), r))
        
        if r > 1 and len(combinaciones) > 50:
            combinaciones = random.sample(combinaciones, 50)

        for combinacion in combinaciones:
            ruta_modificada = ruta.copy()
            ruta_modificada = list(ruta_modificada)

            nodos_bloquados = []
            for i in combinacion:
                nodos_bloquados += list(np.where(cubiertos[i])[0]) + [workers[i][0]]

            if 0 in nodos_bloquados:
                nodos_bloquados.remove(0)

            # Eliminar nodos 
            for node in nodos_bloquados:
                if node in ruta_modificada:
                    ruta_modificada.remove(node)

            # Nuevas distancias
            distancias_modificadas = distancias.copy()
            for node in nodos_bloquados:
                distancias_modificadas[node, :] = float("inf")
                distancias_modificadas[:, node] = float("inf")

            # Reconstruir la ruta
            nueva_ruta = []
            for idx in range(len(ruta_modificada)-1):
                actual = ruta_modificada[idx]
                siguiente_nodo = ruta_modificada[idx+1]
                if distancias_modificadas[actual, siguiente_nodo] == float("inf"):
                    if not nueva_ruta:
                        nueva_ruta = [actual]
                    nueva_ruta = coser_camino(N, adj, actual, siguiente_nodo, nueva_ruta)
                    if nueva_ruta is None:
                        break  # no hay camino posible
                    else: 
                        nueva_ruta = np.array(nueva_ruta, dtype=int).tolist()
                else:
                    if not nueva_ruta:
                        nueva_ruta = [actual]
                    nueva_ruta.append(siguiente_nodo)

            # Calcular nuevo costo
            if not (nueva_ruta is None) and len(nueva_ruta) > 2 and nueva_ruta[0] == 0 and nueva_ruta[-1] == 0:
                nuevo_costo = sum(distancias_modificadas[nueva_ruta[i], nueva_ruta[i+1]] for i in range(len(nueva_ruta)-1))
            else: 
                nuevo_costo = float("inf") 

            if nuevo_costo == float("inf"):
                continue  
            else:
                # Contar trabajadores no cubiertos
                no_cubiertos = sum(
                    not any(cubiertos[k, mp] for mp in nueva_ruta)
                    for k in range(len(workers))
                )

                if no_cubiertos > 0:
                    P = (mejor_costo - nuevo_costo) / no_cubiertos
                    maxP = max(maxP, P)
                    mejor_ruta = nueva_ruta

    print("Ruta encontrada:", mejor_ruta)
    costo = costear_ruta(mejor_ruta, distancias_modificadas)
    if costo == float("inf"):
        print("La ruta no es factible")
    else:
        print("La ruta es es factible")
    return maxP
