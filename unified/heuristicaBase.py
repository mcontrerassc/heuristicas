import time
import numpy as np
from evaluar import dijkstra_multi_source
from heuristica1 import calcular_puntos_de_pickup, heur_grasp
from heuristica2 import construir_camino_greedy, calcular_cubiertos
from auxiliar import mejora_2opt, construir_camino, costear_ruta

# DISTANCIA MÁS CORTA ENTRE todo PAR DE NODOS 
def calcular_distancias(N, adj):
    # calcular todas las distancias más cortas de nodo a nodo
    dist_matrix = np.full((N, N), float('inf'))
    for u in range(N):
        dist = dijkstra_multi_source(N, adj, [u]) 
        for v in range(N):
            dist_matrix[u, v] = dist[v]
    return dist_matrix

# HEURISTICA 1
def heuristica1(N, adj, workers, distancias, iteraciones=200, alpha=0.2, limite=29):
    tiempo_inicial = time.time()
    puntos_pickup = calcular_puntos_de_pickup(workers, distancias)

    mejor_costo = float('inf')
    mejor_ruta = None
    historial = []


    for i in range(iteraciones):
        if time.time() - tiempo_inicial > limite:
            print(f"limite alcanzado en iteracion {i}")
            break
        ruta = heur_grasp(alpha, N, distancias, puntos_pickup, workers)
        ruta, costo = mejora_2opt(ruta, distancias)
        historial.append(costo)
        if costo < mejor_costo:
            mejor_ruta = ruta.copy()
            mejor_costo = costo

    ruta_expandida = construir_camino(N, adj, mejor_ruta)

    return ruta_expandida, mejor_costo, time.time() - tiempo_inicial, historial

# HEURISTICA 2
def heuristica2(N, adj, workers, distancias, iteraciones=1000, limite=29):
    tiempo_inicial = time.time()
    i = 0
    best_route = None
    best_cost =  float('inf')
    historial = []
    distanciasWaN = distancias[[w[0] for w in workers], :]  # (W x N) distancia worker->nodo
    radios = np.array([w[1] for w in workers], dtype=float).reshape(-1, 1)
    cubiertos = calcular_cubiertos(distanciasWaN, radios)  # (W x N) booleano worker->cubre nodo
    while i < iteraciones:
        if time.time() - tiempo_inicial > limite:
            print(f"limite alcanzado en iteracion {i}")
            break
        ruta, _ = construir_camino_greedy(distancias, cubiertos)
        costo = costear_ruta(ruta, distancias)
        historial.append(costo)

        if costo < best_cost:
            best_cost = costo
            best_route = ruta
        i += 1

    best_route, best_cost = mejora_2opt(best_route, distancias, time_limit=limite - (time.time() - tiempo_inicial))
    ruta_expandida = construir_camino(N, adj, best_route)

    return ruta_expandida, best_cost, time.time() - tiempo_inicial, historial