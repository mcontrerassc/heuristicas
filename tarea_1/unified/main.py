import sys
import pandas as pd
import numpy as np
import random
import re 
import time
from evaluar import load_graph, load_workers, INF 
from heuristicaBase import heuristica1, heuristica2, calcular_distancias


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

    # START TIME 
    start_time = time.time()

    # --------- Grafo ----------
    N, adj = load_graph(rf"..\instancias\{arg1}")

    # --------- Lectura instancia ----------
    workers = load_workers(rf"..\instancias\{arg2}")
    radios = np.array([w[1] for w in workers], dtype=float).reshape(-1, 1)
    
    # --------- Distancias ----------
    distancias = calcular_distancias(N, adj)

    end_time = time.time()

    # --------- Heurísticas ----------
    ruta1, mejor_costo1, time1 = heuristica1(N, adj, workers, distancias, iteraciones=200, alpha=0.2, limite=29)
    print(f"Heuristica 1 finalizó con costo {mejor_costo1} en {time1:.2f} segundos")
    ruta2, mejor_costo2, time2 = heuristica2(N, adj, workers, distancias, iteraciones=1000, limite=29)
    print(f"Heuristica 2 finalizó con costo {mejor_costo2} en {time2:.2f} segundos")
    

    # --------- Best of all routes ----------
    if mejor_costo1 < mejor_costo2:
        ruta = ruta1
        mejor_costo = mejor_costo1
        print(f"Heuristica 1 mejor con costo {mejor_costo1}")
    else:
        ruta = ruta2
        mejor_costo = mejor_costo2
        print(f"Heuristica 2 mejor con costo {mejor_costo2}")

    # END TIME 
    final_time = time1 + time2 + (end_time - start_time)

    print(f"Execution time: {final_time:.2f} seconds")

    # --------- Write solution ----------
    with open(fr'..\instancias\solucion{idx}.txt', 'w') as f:
        f.write(' '.join(map(str, ruta)))