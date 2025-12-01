import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from GRASP_Genetico import redistricting, set_global_seed, asignar_escaños, calcular_polsby_popper, calcular_reock


RUTA_SF = "/Users/michellecontreras/heuristicas_tareas/heuristicas/proyecto_final/alaska_precincts.shp"
POP_COL = "POPULATION"     
NUM_DIST = 20
NUM_SEATS = 20            
RANDOM_SEED = 42


def calcular_map_y_compacidad(gdf, assignment, pop_col, num_dist, num_seats):
    """
    assignment: array/list de largo len(gdf) con el id de distrito (0..num_dist-1) de cada unidad.
    MAP: se calcula con la misma definición que en la heurística:
         MAP = 0.5 * sum_i |E_i - P_i|
         donde P_i = frac. de población, E_i = frac. de escaños.

    Además calcula:
      - Polsby–Popper promedio y mínimo
      - Reock promedio y mínimo

    RETURN:
      map_score, avg_pp, min_pp, avg_reock, min_reock
    """
    assignment = np.array(assignment)

    distritos_data = []
    for d in range(num_dist):
        mask = (assignment == d)
        poblacion = float(gdf.loc[mask, pop_col].sum())
        distritos_data.append({
            "Poblacion": poblacion,
            "Escaños": 0
        })

    escaños = asignar_escaños(distritos_data, num_seats)

    total_poblacion = sum(d["Poblacion"] for d in distritos_data)
    if total_poblacion == 0 or num_seats == 0:
        map_score = 1.0 
    else:
        map_acc = 0.0
        for i, d in enumerate(distritos_data):
            P_i = d["Poblacion"] / total_poblacion
            E_i = escaños[i] / num_seats
            map_acc += abs(E_i - P_i)
        map_score = 0.5 * map_acc

    compacidades_pp = []
    reocks = []

    for d in range(num_dist):
        sub = gdf[assignment == d]
        if sub.empty:
            compacidades_pp.append(0.0)
            reocks.append(0.0)
            continue
        geom_union = sub.unary_union
        compacidades_pp.append(calcular_polsby_popper(geom_union))
        reocks.append(calcular_reock(geom_union))

    compacidad_media = float(np.mean(compacidades_pp))   # Avg. Polsby–Popper
    min_pp = float(np.min(compacidades_pp))              # Min. Polsby–Popper
    avg_reock = float(np.mean(reocks))                   # Avg. Reock
    min_reock = float(np.min(reocks))                    # Min. Reock

    return map_score, compacidad_media, min_pp, avg_reock, min_reock



def main():
    gdf = gpd.read_file(RUTA_SF).reset_index(drop=True)

    # A) PLAN ALEATORIO
    np.random.seed(RANDOM_SEED)
    asig_random = np.random.randint(0, NUM_DIST, size=len(gdf))

    (
        map_rand,
        comp_rand,          # Avg. Polsby–Popper
        min_pp_rand,        # Min. Polsby–Popper
        avg_reock_rand,     # Avg. Reock
        min_reock_rand      # Min. Reock
    ) = calcular_map_y_compacidad(
        gdf, asig_random, POP_COL, NUM_DIST, NUM_SEATS
    )

    print("=== Plan aleatorio (20 distritos) ===")
    print(f"MAP aleatorio:              {map_rand:.4f}")
    print(f"Polsby–Popper promedio:     {comp_rand:.43f}")
    print(f"Polsby–Popper mínimo:       {min_pp_rand:.4f}")
    print(f"Reock promedio:             {avg_reock_rand:.4f}")
    print(f"Reock mínimo:               {min_reock_rand:.4f}")

    # PLAN HEURÍSTICO (GRASP+AG)
    set_global_seed(RANDOM_SEED)

    best_plan, seats_heur, map_internal = redistricting(
        shapefile_path=RUTA_SF,
        num_distritos=NUM_DIST,
        num_escaños=NUM_SEATS,
        num_iteraciones_grasp=5,
        num_generaciones_ag=5,
        cant_poblacion_ag=10,
        grasp_alpha=0.2
    )


    colors = plt.cm.get_cmap('Spectral', NUM_DIST) 

    asig_heur = -np.ones(len(gdf), dtype=int)
    for d, distrito in enumerate(best_plan):
        for u in distrito:
            asig_heur[int(u)] = d

    (
        map_heur,
        comp_heur,
        min_pp_heur,
        avg_reock_heur,
        min_reock_heur
    ) = calcular_map_y_compacidad(
        gdf, asig_heur, POP_COL, NUM_DIST, NUM_SEATS
    )

    print("\n=== Plan heurístico (GRASP+AG, 20 distritos) ===")
    print(f"MAP heurístico:             {map_heur:.4f}")
    print(f"Polsby–Popper promedio:     {comp_heur:.4f}")
    print(f"Polsby–Popper mínimo:       {min_pp_heur:.4f}")
    print(f"Reock promedio:             {avg_reock_heur:.4f}")
    print(f"Reock mínimo:               {min_reock_heur:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plan aleatorio
    gdf_random = gdf.copy()
    gdf_random["distrito"] = asig_random
    gdf_random.plot(
        column="distrito",
        ax=axes[0],
        edgecolor="black",
        linewidth=0.2,
        legend=False,
        cmap=colors
    )

    axes[0].set_axis_off()
    axes[0].set_title(
        f"Aleatorio – {NUM_DIST} distritos\n"
        f"MAP = {map_rand:.4f}   |   PP̄ = {comp_rand:.3f}   |   Reock̄ = {avg_reock_rand:.3f}"
    )

    # Plan heurístico
    gdf_heur = gdf.copy()
    gdf_heur["distrito"] = asig_heur
    gdf_heur.plot(
        column="distrito",
        ax=axes[1],
        edgecolor="black",
        linewidth=0.2,
        legend=False,
        cmap=colors
    )

    axes[1].set_axis_off()
    axes[1].set_title(
        f"Heurística (GRASP+AG) – {NUM_DIST} distritos\n"
        f"MAP = {map_heur:.4f}   |   PP̄ = {comp_heur:.3f}   |   Reock̄ = {avg_reock_heur:.3f}"
    )

    plt.tight_layout()
    plt.savefig("alaska_20_random_vs_heuristica_numeros.png", dpi=300)
    plt.show()



if __name__ == "__main__":
    main()
