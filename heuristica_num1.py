import networkx as nx
import random
import time
import numpy as np
from evaluar import dijkstra_multi_source


def calcular_caminos_mas_cortos(N, adj):
    # calcular todas las distancias más cortas de nodo a nodo
    dist_matrix = np.full((N, N), float('inf'))
    for u in range(N):
        dist = dijkstra_multi_source(N, adj, [u]) 
        for v in range(N):
            dist_matrix[u, v] = dist[v]
    return dist_matrix


def calcular_puntos_de_pickup(trabajadores, distancias):
    # compute todos los puntos pickup validos para cada trabajador
    puntos_pickup = {}
    for indice, (casa, radio) in enumerate(trabajadores):
        puntos_validos = set()
        # reivsamos todos los nodos
        for nodo in range(distancias.shape[0]):
            if distancias[casa, nodo] <= radio:
                puntos_validos.add(nodo)
        puntos_pickup[indice] = puntos_validos
    return puntos_pickup


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


def heur_grasp(alpha, N, distancias, puntos_pickup, trabajadores):
    ruta = []
    trabajadores_sin_recoger = set(range(len(trabajadores)))
    nodo_actual = 0

    while trabajadores_sin_recoger:
        # opciones viable: nodos que pueden atender trabajadores sin recoger
        opciones = []
        for nodo in range(N):
            if nodo == nodo_actual:
                continue
            # contar cuántos trabajadores no recogidos pueden caminar hasta este punto
            trabajadores_recogidos = set()
            for indice in trabajadores_sin_recoger:
                # si el nodo esta en los puntos pickup a los que puede llegar este trabajdor
                if nodo in puntos_pickup[indice]:
                    trabajadores_recogidos.add(indice)

            if trabajadores_recogidos:
                distancia = distancias[nodo_actual, nodo]
                if distancia < float('inf'):
                    # priorizar nodos qque sirven mas trabajadores con menor costo de ruta
                    score = len(trabajadores_recogidos) / (1 + distancia)
                    # guardar cada nodo con su score y num de trabajadores recogidos
                    opciones.append((nodo, score, trabajadores_recogidos))

        if not opciones:
            break

        # ordenar las opciones de forma descendiente (mayor es mejor)
        opciones.sort(key=lambda tup: tup[1], reverse=True)

        # guardar el mejor score, permitiremos explorar opciones que se desvien por alpha (bajo score)
        mejor_numero = opciones[0][1]
        lower_bound = mejor_numero - (mejor_numero * alpha)
        rcl = []
        for c in opciones:
            if c[1] >= lower_bound:
                rcl.append(c)

        nodo_escogido, _, trabajadores_recogidos = random.choice(rcl)
        ruta.append(nodo_escogido)
        nodo_actual = nodo_escogido
        trabajadores_sin_recoger -= trabajadores_recogidos

    return ruta


def mejora_2opt(ruta, distancias):
    if len(ruta) < 2:
        return ruta

    mejorada = True
    mejor_ruta = ruta.copy()
    mejor_costo = costear_ruta(ruta, distancias)

    while mejorada:
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

    return mejor_ruta


def expandir_ruta(ruta, grafo):
    # una vez tenemos los puntos pickup, calculamos la ruta completa
    ruta_final = []
    actual = 0

    for siguiente_pickup in ruta:
        segmento = nx.shortest_path(grafo, source=actual, target=siguiente_pickup, weight="weight")
        # ignorar el primer punto, ya que estaba incluido en el segment atnerior
        if ruta_final:
            ruta_final.extend(segmento[1:])
        else:
            ruta_final.extend(segmento)
        actual = siguiente_pickup

    # segmento final para volver a la empresa
    segmento = nx.shortest_path(grafo, source=actual, target=0, weight="weight")
    ruta_final.extend(segmento[1:])
    return ruta_final


def resolver(iteraciones, alpha, limite, k, grafo, N, adj, trabajadores):
    tiempo_inicial = time.time()
    distancias = calcular_caminos_mas_cortos(N, adj)
    puntos_pickup = calcular_puntos_de_pickup(trabajadores, distancias)

    mejor_costo = float('inf')
    mejor_ruta = None

    for i in range(iteraciones):
        if time.time() - tiempo_inicial > limite:
            print(f"limite alcanzado en iteracion {i}")
            break
        ruta = heur_grasp(alpha, N, distancias, puntos_pickup, trabajadores)
        ruta = mejora_2opt(ruta, distancias)
        costo = costear_ruta(ruta, distancias)

        if costo < mejor_costo:
            mejor_ruta = ruta.copy()
            mejor_costo = costo

    ruta_expandida = expandir_ruta(mejor_ruta, grafo)
    # guardar en archivo
    with open(f"solucion{k}.txt", "w") as f:
        f.write(" ".join(map(str, ruta_expandida)))
    return ruta_expandida
