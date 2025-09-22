import sys
import numpy as np
import re 
import time
from evaluar import load_graph, load_workers 
from heuristicaBase import heuristica1, heuristica2, calcular_distancias
from auxiliar import evaluate_exclusion_combinations

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
    ruta2, mejor_costo2, time2, cubiertos = heuristica2(N, adj, workers, distancias, iteraciones=1000, limite=29)
    print(f"Heuristica 2 finalizó con costo {mejor_costo2} en {time2:.2f} segundos")
    

    # --------- Best of all routes ----------
    if ruta1 is None and ruta2 is None:
        print("Ninguna heurística encontró una solución válida.")
        sys.exit(1)
    elif ruta1 is None:
        ruta = ruta2
        mejor_costo = mejor_costo2
        print(f"Heuristica 2 mejor con costo {mejor_costo2}")
        # --------- Evaluate exclusion of workers ----------
        P2 = evaluate_exclusion_combinations(N, adj, distancias, workers, cubiertos, ruta2, mejor_costo2, r=6)
        print(f"El mejor P logrado es: {P2:.4f} con la Heuristica 2 \n")
    elif ruta2 is None:
        ruta = ruta1
        mejor_costo = mejor_costo1
        print(f"Heuristica 1 mejor con costo {mejor_costo1}")
        # --------- Evaluate exclusion of worker ----------
        P1 = evaluate_exclusion_combinations(N, adj, distancias, workers, cubiertos, ruta1, mejor_costo1, r=6)
        print(f"El mejor P logrado es: {P1:.4f} con la Heuristica 1 \n")
    else:
        if mejor_costo1 < mejor_costo2:
            ruta = ruta1
            mejor_costo = mejor_costo1
            print(f"Heuristica 1 mejor con costo {mejor_costo1}")
        else:
            ruta = ruta2
            mejor_costo = mejor_costo2
            print(f"Heuristica 2 mejor con costo {mejor_costo2}")

        # --------- Evaluate exclusion of workers ----------

        P1 = evaluate_exclusion_combinations(N, adj, distancias, workers, cubiertos, ruta1, mejor_costo1, r=6)
        print(f"Valor máximo P a partir de la Heuristica 1 es: {P1:.4f}")

        P2 = evaluate_exclusion_combinations(N, adj, distancias, workers, cubiertos, ruta2, mejor_costo2, r=6)
        print(f"Valor máximo P a partir de la Heuristica 2 es: {P2:.4f}")

        if P1 > P2:
            print(f"El mejor P logrado es: {P1:.4f} con la Heuristica 1 \n")
        else: 
            print(f"El mejor P logrado es: {P2:.4f} con la Heuristica 2 \n")

    # END TIME 
    final_time = time1 + time2 + (end_time - start_time)

    print(f"Execution time: {final_time:.2f} seconds \n")


    # --------- Write solution ----------
    with open(fr'..\instancias\solucion{idx}.txt', 'w') as f:
        f.write(' '.join(map(str, ruta)))
