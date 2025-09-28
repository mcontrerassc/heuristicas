import matplotlib.pyplot as plt
import numpy as np
import os
import random
from evaluar import load_graph, load_workers
from heuristicaBase import heuristica1, heuristica2, calcular_distancias

def plot_best_so_far(hist, label, instancia_idx, outdir="."):
    best_so_far = np.minimum.accumulate(hist)
    plt.figure(figsize=(8,5))
    plt.plot(hist, alpha=0.3, label=f"{label} (original)")
    plt.plot(best_so_far, color="blue", linewidth=2, label=f"{label} (mejor hasta ahora)")

    # marcar el mejor costo
    best_idx = np.argmin(hist)
    best_cost = hist[best_idx]
    plt.scatter(best_idx, best_cost, color="red")
    plt.text(best_idx, best_cost, f"{best_cost:.1f}",
             ha="left", va="bottom", fontsize=9, color="red")

    plt.xlabel("Iteración")
    plt.ylabel("Costo de la ruta")
    plt.title(f"{label} - Instancia {instancia_idx}")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(f"{outdir}/{label}_instancia{instancia_idx}.png")
    plt.close()

if __name__ == "__main__":
    # mismo seed que usamos en heuristica base
    #SEED = 123456
    #random.seed(SEED)
    #np.random.seed(SEED)
    base_dir = "../instancias"
    gpath = os.path.join(base_dir, "grafo.csv")
    outdir = "graficos"
    os.makedirs(outdir, exist_ok=True)

    N, adj = load_graph(gpath)
    distancias = calcular_distancias(N, adj)

    for idx in range(1, 11):
        workers_path = os.path.join(base_dir, f"instancia{idx}.csv")

        print(f"Ploteo instancia {idx}...")
        workers = load_workers(workers_path)

        ruta1, cost1, tiempo1, hist1 = heuristica1(N, adj, workers, distancias)
        ruta2, cost2, tiempo2, hist2 = heuristica2(N, adj, workers, distancias)

        plot_best_so_far(hist1, "Heuristica 1", idx, outdir=outdir)
        plot_best_so_far(hist2, "Heuristica 2", idx, outdir=outdir)

        print(f"Instancia {idx}, Costos: H1={cost1:.1f}, H2={cost2:.1f}")

