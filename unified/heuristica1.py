import random 
import numpy as np
SEED = 123456
random.seed(SEED)
np.random.seed(SEED)

def calcular_puntos_de_pickup(workers, distancias):
    # compute todos los puntos pickup validos para cada trabajador
    puntos_pickup = {}
    for indice, (casa, radio) in enumerate(workers):
        puntos_validos = set()
        # reivsamos todos los nodos
        for nodo in range(distancias.shape[0]):
            if distancias[casa, nodo] <= radio:
                puntos_validos.add(nodo)
        puntos_pickup[indice] = puntos_validos
    return puntos_pickup

def heur_grasp(alpha, N, distancias, puntos_pickup, trabajadores):
    k = 10
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
                # limitar tamaño máximo de la RCL
        if k is not None and len(rcl) > k:
            rcl = rcl[:k]

        nodo_escogido, _, trabajadores_recogidos = random.choice(rcl)
        ruta.append(nodo_escogido)
        nodo_actual = nodo_escogido
        trabajadores_sin_recoger -= trabajadores_recogidos

    return ruta

