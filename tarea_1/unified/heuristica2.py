import numpy as np
from auxiliar import coser_camino
from evaluar import dijkstra_multi_source

rng = np.random.default_rng(123456)

# --------- Covertura de workers por nodos ----------
def calcular_cubiertos(distanciasWaN, radios):
    """
    distanciasWaN: (W x N) distancia worker->nodo
    radios: (W,) radio por worker
    retorna: (W x N) tal que la pos [i, j]==1 si distanciasWaN[i,j] < radios[i] (worker i es cubierto por nodo j)
    """
    radios = np.asarray(radios, float).reshape(-1, 1)
    return np.isfinite(distanciasWaN) & (distanciasWaN < radios)  # bool (W x N)

# --------- Armado de ruta ----------
def armado_greedy(distancias, nodes):
    nodes = [int(x) for x in nodes]
    if not nodes:
        return np.array([0, 0], dtype=int)

    # 1) elegir 'first' como el nodo más cercano a 0 entre los alcanzables
    d0 = distancias[0, :] # distancias desde 0 a todos
    reachable = [n for n in nodes if np.isfinite(d0[n])]
    if reachable:
        first = min(reachable, key=lambda x: d0[x])
    else:
        # si ninguno es alcanzable desde 0, no existe ruta factible que empiece en 0
        # podés lanzar error o elegir igual (fallará generate_minimal_route si no hay camino)
        raise ValueError("Ninguno de los nodos seleccionados es alcanzable desde 0; no hay ruta factible.")

    remaining = [n for n in nodes if n != first]

    # 2) ruta inicial: el CAMINO MÍNIMO 0 -> first
    #route = np.array([0], dtype=int)
    route = [0]

    # 3) greedy nearest-neighbor cosiendo SIEMPRE el camino mínimo entre consecutivos
    prev = first
    while remaining:
        dprev = distancias[prev, :]
        nxt = min(remaining, key=lambda x: dprev[x])
        if not np.isfinite(dprev[nxt]):
            raise ValueError(f"No hay camino desde {prev} hasta {nxt}; grafo desconectado.")
        route.append(nxt)
        prev = nxt
        remaining.remove(nxt)
        #print(remaining)

    route.append(0)  # volver a 0 al final

    return route

# --------- Selección de nodos claves ----------
def select_nodes(binario, umbral):
    """
    Selecciona nodos que cubren a más de un umbral workers para que formen parte de la ruta.
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

# --- Función Principal ---
def construir_camino_greedy(distancias, cubiertos):
    """
    distancias: (N x N) distancia worker->nodo
    cubiertos: (W x N) booleano worker->cubre nodo
    """
    # Elegimos umbral random entre 3 y la máxima cantidad de workers cubiertos por un nodo
    m = np.max(cubiertos.sum(axis=0))
    #print(f"max cubiertos por nodo: {m}")
    if m < 3:
        umbral = m
    else:
        umbral = np.random.randint(3,m)

    seleccionados = select_nodes(cubiertos, umbral)
    #print("Nodos seleccionados")

    route = armado_greedy(distancias, seleccionados)
    #print("Ruta armada")
    
    return route, seleccionados


