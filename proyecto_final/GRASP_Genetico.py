import geopandas as gpd
import networkx as nx
import random
import os
import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon # Necesario para geometría avanzada
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import warnings
from shapely.errors import ShapelyDeprecationWarning 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# ==============================================================================
# SEMILLA 
# ==============================================================================

def set_global_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    print(f"La semilla global de aleatoriedad ha sido establecida a: {seed_value}")

set_global_seed(42)

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================
def visualizar_plan(plan_distritos, unidades_gdf, escaños_asignados = None, map_score = None):
    if not plan_distritos or unidades_gdf.empty:
        print("No se puede visualizar: el plan de distritos o el GeoDataFrame está vacío.")
        return

    distritos_gdf = []
    
    colors = plt.cm.get_cmap('Spectral', len(plan_distritos)) 

    for i, distrito_nodes in enumerate(plan_distritos):
        if not distrito_nodes:
            continue

        distrito_subset = unidades_gdf.loc[distrito_nodes]
        
        distrito_geometry = distrito_subset.geometry.unary_union
        
        distritos_gdf.append({
            'ID_Distrito': i + 1,
            'geometry': distrito_geometry,
            'Escaños': escaños_asignados[i] if escaños_asignados else None
        })

    mapa_final_gdf = gpd.GeoDataFrame(distritos_gdf, crs=unidades_gdf.crs)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    mapa_final_gdf.plot(
        column='ID_Distrito', 
        cmap=colors, 
        edgecolor='black', 
        linewidth=0.5,
        legend=False,
        ax=ax
    )

    if escaños_asignados:
        for idx, row in mapa_final_gdf.iterrows():
            centroide = row.geometry.centroid
            ax.annotate(
                text=f"D{row['ID_Distrito']}\n({row['Escaños']} escaños)", 
                xy=(centroide.x, centroide.y),
                ha='center', 
                fontsize=8, 
                color='black',
                weight='bold',
                path_effects=[
                    plt.matplotlib.patheffects.withStroke(linewidth=1.5, foreground="white")
                ]
            )

    title = "Mapa de Distritos Generado por Heurística Híbrida (GRASP + AG)"
    if map_score is not None:
        title += f"\nScore MAP Final: {map_score:.4f} (Menor es Mejor)"

    ax.set_title(title, fontsize=14)
    ax.set_axis_off()
    plt.show()

def calcular_polsby_popper(distrito_geometria):
    if distrito_geometria.area == 0 or distrito_geometria.length == 0:
        return 0.0
    return (4 * np.pi * distrito_geometria.area) / (distrito_geometria.length ** 2)

def calcular_reock(distrito_geometria):
    try:
        minx, miny, maxx, maxy = distrito_geometria.bounds
        bbox_area = (maxx - minx) * (maxy - miny)
        return distrito_geometria.area / bbox_area if bbox_area > 0 else 0.0
    except Exception:
        return 0.5

def asignar_escaños(distritos_data, total_escaños):

    total_poblacion = sum(d['Poblacion'] for d in distritos_data)
    if total_poblacion == 0:
        return [0] * len(distritos_data)

    escaños_asignados = []
    proporciones = [(d['Poblacion'] / total_poblacion) * total_escaños for d in distritos_data]
    
    # Asignación por redondeo
    escaños_base = [min(int(p),1) for p in proporciones]
    escaños_restantes = total_escaños - sum(escaños_base)
    
    restos = [p - b for p, b in zip(proporciones, escaños_base)]
    
    # Asignar los restantes a los distritos con el mayor 'resto'
    indices_mayores_restos = np.argsort(restos)[::-1][:escaños_restantes]
    
    final_escaños = escaños_base[:]
    for i in indices_mayores_restos:
        final_escaños[i] += 1
        
    for i, d in enumerate(distritos_data):
        d['Escaños'] = final_escaños[i]
        
    return final_escaños

def calcular_map(distritos_data, total_escaños):
    if total_escaños == 0: return 1.0 # maximo si no se han asignado escaños
    
    map_score = 0
    total_poblacion = sum(d['Poblacion'] for d in distritos_data)
    
    for distrito in distritos_data:
        P_i = distrito['Poblacion'] / total_poblacion
        E_i = distrito['Escaños'] / total_escaños
        map_score += abs(E_i - P_i)
        
    return 0.5 * map_score

def evaluar_plan(plan_distritos, total_escaños, unidades_gdf, graph):
    # MAXIMIZAR (func_obj = Compacidad Promedio - MAP)

    factible = es_factible(plan_distritos, graph)
    if not factible:
        return -np.inf, np.inf, 0.0, [] 
    distritos_data = []
    
    for unidades_id in plan_distritos:
        distrito_gdf = unidades_gdf.loc[unidades_id]
        
        poblacion_total_distrito = distrito_gdf[POPULATION_COL].sum()
        geometria_distrito = distrito_gdf.geometry.unary_union 
        
        distritos_data.append({
            'Poblacion': poblacion_total_distrito,
            'Geometria': geometria_distrito,
            'Escaños': 0 
        })
    
    asignar_escaños(distritos_data, total_escaños)

    map_score = calcular_map(distritos_data, total_escaños)
    
    pp_scores = [calcular_polsby_popper(d['Geometria']) for d in distritos_data]
    reock_scores = [calcular_reock(d['Geometria']) for d in distritos_data]
    
    compactibilidad_promedio = (np.mean(pp_scores) + np.mean(reock_scores)) / 2

    func_obj = compactibilidad_promedio - map_score 
    
    return func_obj, map_score, compactibilidad_promedio, distritos_data

def es_factible(plan, graph):
    # Todos los nodos deben estar exactamente una vez
    all_nodes = [n for distrito in plan for n in distrito]
    if len(all_nodes) != len(set(all_nodes)):
        return False
    if set(all_nodes) != set(graph.nodes):
        return False
    
    # Ningún distrito vacío
    if any(len(d)==0 for d in plan):
        return False
    
    # Contigüidad de cada distrito
    for distrito in plan:
        sub = graph.subgraph(distrito)
        if not nx.is_connected(sub):
            return False

    return True

def get_district_population(plan_distritos, graph):
    populations = []
    for district_nodes in plan_distritos:
        current_pop = sum(graph.nodes[v]['population'] for v in district_nodes if v in graph.nodes)
        populations.append(current_pop)
    return populations

def reparar_plan(plan_distritos_raw, graph, ideal_pop_target, unidades_gdf=None):
    num_distritos_target = len(plan_distritos_raw)

    # No repetidos 

    assigned_nodes = set()
    plan_clean = [[] for _ in range(num_distritos_target)]
    
    for i, distrito in enumerate(plan_distritos_raw):
        for node in distrito:
            if node not in assigned_nodes:
                plan_clean[i].append(node)
                assigned_nodes.add(node)

    # Todos los nodos deben estar asignados
                
    all_nodes = set(graph.nodes)
    unassigned_nodes = list(all_nodes - assigned_nodes)
    
    for u in unassigned_nodes:
        best_target_j = -1
        # min_pop = np.inf
        best_score = -np.inf
        
        current_pops = get_district_population(plan_clean, graph)
        
        for neighbor in graph.neighbors(u):
            try:
                j = next(idx for idx, d in enumerate(plan_clean) if neighbor in d)
                
                # if current_pops[j] < min_pop:
                #     min_pop = current_pops[j]
                #     best_target_j = j

                nuevos_nodos = plan_clean[j] + [u]
                current_pop_new = current_pops[j] + graph.nodes[u]['population']
                remaining_pop_diff = (ideal_pop_target - current_pop_new) / ideal_pop_target

                distrito_gdf = unidades_gdf.loc[nuevos_nodos]
                geometria_distrito = distrito_gdf.geometry.unary_union

                pp_score = calcular_polsby_popper(geometria_distrito)
                reock_score = calcular_reock(geometria_distrito)
                compactibilidad = (pp_score + reock_score) / 2
                score = compactibilidad - abs(remaining_pop_diff)
                if best_target_j == -1 or score > best_score:
                    best_score = score
                    best_target_j = j

            except StopIteration:
                continue

        # Asignar el nodo 'u' al mejor distrito encontrado
        if best_target_j != -1:
            plan_clean[best_target_j].append(u)
            current_pops[best_target_j] = current_pop_new
        else:
             smallest_district_idx = np.argmin(current_pops)
             plan_clean[smallest_district_idx].append(u)
             current_pops[smallest_district_idx] += graph.nodes[u]['population']


    # Contigüidad 
    
    repaired_units_counter = 1 
    
    while repaired_units_counter > 0:
        repaired_units_counter = 0
        
        # Recorrer cada distrito para verificar la conectividad
        for i in range(num_distritos_target):
            district_nodes = plan_clean[i]
            if len(district_nodes) <= 1: continue

            subgraph = graph.subgraph(district_nodes)
            components = list(nx.connected_components(subgraph))
            
            if len(components) > 1:
                
                # Identificar el más poblado
                comp_pops = [sum(graph.nodes[v]['population'] for v in comp) for comp in components]
                main_component = components[np.argmax(comp_pops)]
                islands = [comp for comp in components if comp != main_component]
                
                for island in islands:
                    nodes_to_move = list(island)
                    
                    # Encontrar el distrito vecino más pequeño para reasignar 
                    neighboring_districts = {}
                    for node in nodes_to_move:
                        for neighbor in graph.neighbors(node):
                            try:
                                j = next(idx for idx, d in enumerate(plan_clean) if neighbor in d)
                                if j != i: neighboring_districts[j] = neighboring_districts.get(j, 0) + 1
                            except StopIteration:
                                continue 
                    
                    current_pops = get_district_population(plan_clean, graph)
                    best_target_j = -1
                    min_pop = np.inf
                    
                    if neighboring_districts:
                        for j in neighboring_districts.keys():
                            if current_pops[j] < min_pop:
                                min_pop = current_pops[j]
                                best_target_j = j
                        
                        if best_target_j != -1:
                            plan_clean[i] = [n for n in plan_clean[i] if n not in nodes_to_move]
                            plan_clean[best_target_j].extend(nodes_to_move)
                            repaired_units_counter += len(nodes_to_move)
                            break 
                
            if repaired_units_counter > 0:
                break 

    # Distritos no Vacíos 

    final_plan = [d for d in plan_clean if d]
    num_empty_districts = num_distritos_target - len(final_plan)

    # Divide el distrito más grande hasta restaurar el número de distritos
    while num_empty_districts > 0:
        
        current_pops = get_district_population(final_plan, graph)
        
        victim_index = np.argmax(current_pops)
        victim_nodes = final_plan[victim_index]
        victim_pop = current_pops[victim_index]

        new_district_pop_target = victim_pop / 2  
        
        seed_node = victim_nodes[0]
        new_district = [seed_node]
        new_district_current_pop = graph.nodes[seed_node]['population']
        
        available_nodes = set(victim_nodes)
        available_nodes.remove(seed_node)
        
        # Mientras el nuevo distrito no alcance su población objetivo O haya nodos adyacentes
        while new_district_current_pop < new_district_pop_target:
            best_candidate = None
            
            # Simplemente encuentra el primer vecino disponible
            for current_node in new_district:
                for neighbor in graph.neighbors(current_node):
                    if neighbor in available_nodes:
                        best_candidate = neighbor
                        break
                if best_candidate: break

            if best_candidate:
                # Añadir el nodo al nuevo distrito
                new_district.append(best_candidate)
                available_nodes.remove(best_candidate)
                new_district_current_pop += graph.nodes[best_candidate]['population']
            else:
                break

        final_plan[victim_index] = [n for n in victim_nodes if n not in new_district]
        
        final_plan.append(new_district)
        
        num_empty_districts -= 1
        
    return final_plan

# ==============================================================================
# GRASP y Algoritmos Genéticos
# ==============================================================================

def grasp_construction(num_distritos, graph, ideal_pop, alpha = 0.2, unidades_gdf=None):
    unassigned_nodes = set(graph.nodes)
    plan_distritos = [[] for _ in range(num_distritos)]
    
    # Asignar una semilla a cada distrito
    for i in range(num_distritos):
        seed = np.random.choice(list(unassigned_nodes))
        unassigned_nodes.remove(seed)
        plan_distritos[i].append(seed)
        
    # Fase de Expansión
    while unassigned_nodes:
        best_candidate = None
        best_score = -np.inf
        target_distrito_idx = -1
        
        # Identificar todas las unidades no asignadas adyacentes a un distrito
        candidate_list = []
        for u in unassigned_nodes:
            for i, distrito_nodes in enumerate(plan_distritos):
                if any(graph.has_edge(u, v) for v in distrito_nodes):

                    current_pop = sum(graph.nodes[v]['population'] for v in distrito_nodes)
                    remaining_pop_diff = (ideal_pop - current_pop) / ideal_pop
                    
                    # distrito_gdf = unidades_gdf.loc[distrito_nodes + [u]]
                    # geometria_distrito = distrito_gdf.geometry.unary_union 
                    # pp_score = calcular_polsby_popper(geometria_distrito)
                    # reock_score = calcular_reock(geometria_distrito)
                    # compactibilidad = (pp_score + reock_score) / 2
                    # score = compactibilidad - abs(remaining_pop_diff)


                    score = remaining_pop_diff                    
                    candidate_list.append((u, i, score))

        if not candidate_list: break # No hay más contiguos
        
        scores = np.array([c[2] for c in candidate_list])
        scores_normalized = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-9)
        
        rcl_threshold = np.max(scores) - alpha * (np.max(scores) - np.min(scores))
        RCL = [c for c in candidate_list if c[2] >= rcl_threshold]
        
        if not RCL: continue
        
        selected_candidate = RCL[np.random.randint(len(RCL))]
        u, i, _ = selected_candidate
        
        unassigned_nodes.remove(u)
        plan_distritos[i].append(u)
        
    return plan_distritos

def grasp_local_search(plan_distritos, graph, unidades_gdf, num_escaños):
    current_func_obj, _, _, _ = evaluar_plan(plan_distritos, num_escaños, unidades_gdf, graph)
    print(f"Func_obj inicial búsqueda local GRASP: {current_func_obj:.4f}")
    improved = True
    improvements = 0
    
    while improved and improvements < 5:
        improved = False
        best_move_plan = plan_distritos
        
        # Iterar sobre todos los nodos y distritos vecinos
        for i in range(len(plan_distritos)): # Distrito de origen
            for j in range(len(plan_distritos)): # Distrito de destino
                if i == j: continue
                
                # Nodos frontera (candidatos a moverse)
                movable_nodes = [
                    u for u in plan_distritos[i] 
                    if any(v in plan_distritos[j] for v in graph.neighbors(u))
                ]
                
                for node_to_move in movable_nodes:
                    # Crear el nuevo plan temporal
                    new_plan = [d[:] for d in plan_distritos]
                    new_plan[i].remove(node_to_move)
                    new_plan[j].append(node_to_move)

                    temp_func_obj, _, _, _ = evaluar_plan(new_plan, num_escaños, unidades_gdf, graph)
                    
                    if temp_func_obj > current_func_obj:
                        current_func_obj = temp_func_obj
                        best_move_plan = new_plan
                        improved = True
                        break # Salir del bucle interno y volver a la búsqueda
                if improved: break
            if improved: break
            
        if improved:
            plan_distritos = best_move_plan
            print(f"- Iteración de búsqueda local #{improvements + 1}/5. Func_obj actual: {current_func_obj:.4f}")
            improvements += 1
            
    return plan_distritos

def genetic_algorithm_crossover(parent1, parent2, graph):
    # Elegir un punto de corte (un distrito de P1 para transferir a P2)
    cut_district_idx = np.random.randint(len(parent1))
    
    # El nuevo hijo hereda ese distrito de P1
    child_plan = [parent1[cut_district_idx]]
    assigned_nodes = set(parent1[cut_district_idx])
    
    # El resto de los nodos se toma de P2 o se reasignan (más complejo)
    for i, dist in enumerate(parent2):
        if i == cut_district_idx: continue
        
        new_dist = [n for n in dist if n not in assigned_nodes]
        if new_dist:
            child_plan.append(new_dist)
            assigned_nodes.update(new_dist)
            
    while len(child_plan) < len(parent1):
        child_plan.append([])
    
    # En una implementación real, se debe reasignar cualquier nodo no asignado
    # y potencialmente dividir/unir distritos para alcanzar D
    return child_plan

# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================

def redistricting(shapefile_path, num_distritos, num_escaños, 
                         num_iteraciones_grasp = 20, num_generaciones_ag = 5, cant_poblacion_ag = 50, 
                         grasp_alpha = 0.2):

    # GRASP + Algoritmos Genéticos

    print("Iniciando Heurística Híbrida para Redistricting...")
    
    try:
        unidades_gdf = gpd.read_file(shapefile_path)
        if unidades_gdf.crs.is_geographic:
             unidades_gdf = unidades_gdf.to_crs(unidades_gdf.estimate_utm_crs()) 
    except Exception as e:
        print(f"Error al leer shapefile o CRS: {e}")
        return None, None, None
        
    graph = nx.Graph()
    total_poblacion_global = unidades_gdf[POPULATION_COL].sum()
    ideal_pop_target = total_poblacion_global / num_distritos
    
    # Añadir nodos con peso (población)
    for index, row in unidades_gdf.iterrows():
        graph.add_node(index, population=row[POPULATION_COL])
        
    # Aristas 
    for i in unidades_gdf.index:
        for j in unidades_gdf.index:
            if i < j and unidades_gdf.loc[i].geometry.touches(unidades_gdf.loc[j].geometry):
                graph.add_edge(i, j)

    print(f"   Total de Unidades: {len(unidades_gdf)}. Población ideal por distrito: {ideal_pop_target:.0f}")
    
    # Fase de Construcción: GRASP  
    print("\n Fase de Construcción (GRASP) \n")
    soluciones_grasp = []
    
    for _ in range(num_iteraciones_grasp):
        # Construcción
        print("Generando solución inicial GRASP...")
        plan_initial = grasp_construction(num_distritos, graph, ideal_pop_target, grasp_alpha, unidades_gdf)
        
        # Búsqueda Local
        print("Mejorando solución con búsqueda local...")
        plan_optimizado = grasp_local_search(plan_initial, graph, unidades_gdf, num_escaños)
        
        # Evaluación
        print("\n Evaluando solución GRASP...")
        func_obj, map_score, compacidad, distritos_data = evaluar_plan(plan_optimizado, num_escaños, unidades_gdf, graph)
        soluciones_grasp.append({'plan': plan_optimizado, 'func_obj': func_obj, 'map': map_score, 'compacidad': compacidad})
        
        print(f"    FIN Iteración GRASP {_+1}/{num_iteraciones_grasp}: func_obj: {func_obj:.4f}, MAP: {map_score:.4f}, Compacidad: {compacidad:.4f} \n")

    soluciones_grasp.sort(key=lambda x: x['func_obj'], reverse=True)
    print(f"   Generadas {len(soluciones_grasp)} soluciones iniciales. Mejor func_obj GRASP: {soluciones_grasp[0]['func_obj']:.4f}")
    
    # Fase de Mejora: Algoritmo Genético 
    print("\n Fase de Mejora (Algoritmo Genético)...")
    
    # Población inicial: las mejores soluciones de GRASP
    poblacion_ag = [s['plan'] for s in soluciones_grasp[:num_iteraciones_grasp]] 
    mejor_plan_global = soluciones_grasp[0]['plan']
    mejor_func_obj = soluciones_grasp[0]['func_obj']

    generaciones_sin_mejora = 0
    
    for gen in range(num_generaciones_ag):
        print(f"\n Generación {gen+1}/{num_generaciones_ag} \n")
        func_objs = [evaluar_plan(plan, num_escaños, unidades_gdf, graph)[0] for plan in poblacion_ag]

        mejores_indices = np.argsort(func_objs)[::-1]
        nueva_poblacion = [poblacion_ag[i] for i in mejores_indices[:5]] # Usar las N mejores
        
        while len(nueva_poblacion) < cant_poblacion_ag:
            print("Creando nuevo individuo mediante crossover y mutación...")
            # Selección de dos padres de los mejores
            p1_idx, p2_idx = np.random.choice(mejores_indices[:10], size=2, replace=False)
            parent1 = poblacion_ag[p1_idx]
            parent2 = poblacion_ag[p2_idx]
            
            child = genetic_algorithm_crossover(parent1, parent2, graph)
            child = reparar_plan(child, graph, ideal_pop_target, unidades_gdf)
            #nueva_poblacion.append(child)

            child_mutated = grasp_local_search(child, graph, unidades_gdf, num_escaños) 
            nueva_poblacion.append(child_mutated)
            
        poblacion_ag = nueva_poblacion
        
        current_best_func_obj = np.max(func_objs)
        if current_best_func_obj > mejor_func_obj:
            mejor_func_obj = current_best_func_obj
            mejor_plan_global = poblacion_ag[np.argmax(func_objs)]
            generaciones_sin_mejora = 0
        else:
            generaciones_sin_mejora += 1
        
            if generaciones_sin_mejora >= PACIENCIA:
                print(f" No hubo mejora durante {PACIENCIA} generaciones consecutivas.")
                break 
            
        print(f"   Gen {gen+1}: Mejor func_obj: {mejor_func_obj:.4f}")

    final_func_obj, final_map, final_compacidad, distritos_data = evaluar_plan(mejor_plan_global, num_escaños, unidades_gdf, graph)
    escaños_asignados = [d['Escaños'] for d in distritos_data]
    
    print(f"   MAP final: {final_map:.4f} (Objetivo: Minimizar)")
    print(f"   Compacidad promedio final: {final_compacidad:.4f} (Objetivo: Maximizar)")
    
    return mejor_plan_global, escaños_asignados, final_map


RUTA_SHAPEFILE = r"C:\Users\noefa\Desktop\Facultad\Heuristicas\heuristicas\proyecto_final\base_santa_fe.shp"
POPULATION_COL = 'poblacion' 
D_DISTRICTS = 15 
S_SEATS = 70
NUM_ITERACIONES_GRASP = 30  # número de soluciones GRASP a generar
NUM_GENERACIONES_AG = 5 # numero de veces que se ejecuta el algoritmo genético completo (toma las 5 mejores de las generadas por grasp y genera 15 nuevas)
CANT_POBLACION_AG = 15 # Tamaño final de la población en AG
PACIENCIA = 6 # Número de generaciones sin mejora

best_plan, seats, final_map_score = redistricting(
    shapefile_path=RUTA_SHAPEFILE, 
    num_distritos=D_DISTRICTS, 
    num_escaños=S_SEATS, 
    num_iteraciones_grasp=NUM_ITERACIONES_GRASP, 
    num_generaciones_ag=NUM_GENERACIONES_AG, 
    cant_poblacion_ag=CANT_POBLACION_AG
)

print(f"\nMejor Plan (Índices de Unidades por Distrito): {best_plan}")
print(f"Escaños Asignados a cada Distrito: {seats}")

unidades_gdf = gpd.read_file(RUTA_SHAPEFILE)
visualizar_plan(
    plan_distritos=best_plan, 
    unidades_gdf=unidades_gdf, 
    escaños_asignados=seats, 
    map_score=final_map_score
)