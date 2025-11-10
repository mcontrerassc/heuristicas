import random
import gurobipy as gp
from gurobipy import GRB
import time
import collections

# ---------------------------- FP ----------------------------

def clone_as_lp(model, xvars):
    """
    Clona el modelo como LP: todas las variables pasan a continuas.
    Devuelve (lp, var_map) donde var_map: original_var -> cloned_var
    """
    lp = model.copy()
    lp.Params.OutputFlag = 0
    lp.Params.Presolve = 2
    # vuelve todo continuo
    for v in lp.getVars():
        v.VType = GRB.CONTINUOUS
    lp.update()

    # Pequeño arreglo por si los índices no coinciden
    name_to_lpvar = {v.VarName: v for v in lp.getVars()}
    var_map = {v: name_to_lpvar[v.VarName] for v in xvars}
    return lp, var_map

def build_distance_layer(lp, x_lp_vars, name_prefix="fp"):
    """
    Crea variables d_j >= |x_j - z_j| linealizado:
        d_j - x_j >= -z_j
        d_j + x_j >=  z_j
    (z_j es constante que se actualiza en RHS cada iteración)
    Devuelve (d_vars, cons1, cons2), donde cons1,cons2 son listas de constrs por j.
    """
    d_vars = []
    cons1 = []  # d_j - x_j >= -z_j   (RHS = -z_j)
    cons2 = []  # d_j + x_j >=  z_j   (RHS =  z_j)
    for j, xj in enumerate(x_lp_vars):
        dj = lp.addVar(lb=0.0, name=f"{name_prefix}_d[{j}]")
        d_vars.append(dj)
        c1 = lp.addConstr(
            dj - xj >= 0.0, name=f"{name_prefix}_c1[{j}]"
        )  # RHS se ajustará
        c2 = lp.addConstr(
            dj + xj >= 0.0, name=f"{name_prefix}_c2[{j}]"
        )  # RHS se ajustará
        cons1.append(c1)
        cons2.append(c2)
    lp.update()
    # Objetivo: min sum d_j  (lo actualizaremos si quieres ponderar con el original)
    lp.setObjective(gp.quicksum(d_vars), GRB.MINIMIZE)
    lp.update()
    return d_vars, cons1, cons2

def update_z_rhs(lp, cons1, cons2, z):
    """
    Actualiza RHS de:
      d - x >= -z  (RHS = -z)
      d + x >=  z  (RHS =  z)
    """
    for j in range(len(z)):
        cons1[j].RHS = -float(z[j])
        cons2[j].RHS = float(z[j])
    lp.update()

def round_ints(x_vals, vtypes, eps=1e-6, seed=None):
    """
    Redondeo "cercano" para enteras y binaria; tie-break aleatorio si muy cerca de .5
    Devuelve z (solo para variables enteras/binary; para continuas usamos x_vals
    """
    if seed is not None:
        random.seed(seed)
    z = [None] * len(x_vals)
    for j, (xj, t) in enumerate(zip(x_vals, vtypes)):
        if t in (GRB.BINARY, GRB.INTEGER):
            r = round(xj)
            # tie-break suave alrededor de .5
            if abs(xj - 0.5) <= 1e-9 and t == GRB.BINARY:
                r = 1 if random.random() < 0.5 else 0
            z[j] = float(r)
        else:
            z[j] = float(xj)
    return z

def distance_L1(x_vals, z, mask_int):
    """||x - z||_1 sobre componentes enteras/binarias (mask_int=True)."""
    return sum(abs(x_vals[j] - z[j]) for j in range(len(x_vals)) if mask_int[j])

def perturb_z(z, vtypes, num_flips=1, frac_order=None, seed=None):
    """
    Anticiclo: voltea unas pocas variables (enteras/binary).
    Si se provee frac_order (índices ordenados por fraccionalidad previa), usa esos.
    """
    if seed is not None:
        random.seed(seed)
    idxs = [j for j, t in enumerate(vtypes) if t in (GRB.BINARY, GRB.INTEGER)]
    if not idxs:
        return z[:]
    candidates = frac_order if frac_order is not None else idxs
    z_new = z[:]
    flips = 0
    for j in candidates:
        if vtypes[j] == GRB.BINARY:
            z_new[j] = 1.0 - z_new[j]
        else:
            # para enteras, +1 o -1 aleatorio
            z_new[j] = z_new[j] + (1 if random.random() < 0.5 else -1)
        flips += 1
        if flips >= num_flips:
            break
    return z_new

def check_feasible_by_fixing_integers(orig_model, xvars, z_int, time_limit=5.0):
    """
    Verifica factibilidad fuerte: clona modelo original, fija enteras a z_int,
    optimiza como LP factible (objetivo 0).
    Devuelve (is_feasible, sol_values_dict) si factible.
    """
    mm = orig_model.copy()
    mm.Params.OutputFlag = 0
    mm.Params.TimeLimit = time_limit
    # Fijar enteras
    name_to_var = {v.VarName: v for v in mm.getVars()}
    for v, val in zip(xvars, z_int):
        w = name_to_var[v.VarName]
        if w.VType in (GRB.BINARY, GRB.INTEGER):
            w.LB = w.UB = float(val)
        else:
            # continuas no se fijan
            pass
    mm.update()
    # objetivo 0
    mm.setObjective(0.0, GRB.MINIMIZE)
    mm.optimize()
    if mm.status == GRB.OPTIMAL or mm.status == GRB.SUBOPTIMAL:
        # extraer solución
        sol = {var.VarName: var.X for var in mm.getVars()}
        return True, sol
    return False, {}

def feasibility_pump_seeded(
    model,
    xvars,
    x0,  # semilla inicial (solución LP relajada del callback)
    max_iters=40,  # número máximo de iteraciones FP
    time_limit=10.0,  # tiempo máximo total (segundos)
    int_eps=1e-6,  # tolerancia para redondeo de enteras
    seed=0,  # semilla aleatoria
    verbose=False,
):  # imprimir info detallada o no
    # Marca de tiempo inicial (para cortar por tiempo)
    t0 = time.time()
    # Inicializa semilla aleatoria (para redondeo y perturbaciones)
    random.seed(seed)

    # Tipos de variables (BINARY, INTEGER o CONTINUOUS)
    vtypes = [v.VType for v in xvars]
    # Máscara booleana: True si variable es entera o binaria
    mask_int = [t in (GRB.BINARY, GRB.INTEGER) for t in vtypes]

    # === 1) Construcción del LP auxiliar para la proyección ===
    # Clona el modelo original pero cambia todas las variables a continuas
    lp, var_map = clone_as_lp(model, xvars)
    # Obtiene las variables clonadas en el mismo orden
    x_lp_vars = [var_map[v] for v in xvars]
    # Añade una capa de “distancia” (d_j >= |x_j - z_j|) que se actualizará en cada iteración
    d_vars, cons1, cons2 = build_distance_layer(lp, x_lp_vars, name_prefix="fp")

    # === 2) Inicialización ===
    # Redondeo inicial: convierte la solución LP (x0) en entera z
    z = round_ints(x0, vtypes, eps=int_eps, seed=seed)
    # Calcula la distancia L1 inicial entre x0 y z (solo en componentes enteras)
    best_L1 = distance_L1(x0, z, mask_int)
    if verbose:
        print(f"[FP-seeded] it=0  ||x-z||_1(int)={best_L1:.6g}")

    # Actualiza los RHS (lados derechos) de las restricciones d - x >= -z y d + x >= z
    update_z_rhs(lp, cons1, cons2, z)

    # Guarda los redondeos anteriores para evitar ciclos
    z_history = set()
    # Convierte z a tupla hashable (solo enteras) para registrar la configuración actual
    z_key = tuple(int(zj) if mask_int[j] else 0 for j, zj in enumerate(z))
    z_history.add(z_key)

    # Contadores de iteración y sin mejora
    it, no_improve = 0, 0

    # === 3) Bucle principal del Feasibility Pump ===
    while it < max_iters and (time.time() - t0) < time_limit:
        it += 1

        # (a) Minimiza la distancia ||x - z||_1 en el LP
        lp.optimize()
        # Si el LP no es óptimo o factible → se detiene
        if lp.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            if verbose:
                print(f"[FP-seeded] LP infactible en it={it}.")
            break

        # (b) Extrae la solución relajada x (continuas)
        x_rel = [xj.X for xj in x_lp_vars]
        # (c) Redondea nuevamente x_rel → z_new
        z_new = round_ints(x_rel, vtypes, eps=int_eps, seed=seed + it)
        # (d) Calcula nueva distancia ||x - z||_1 sobre enteras
        L1 = distance_L1(x_rel, z_new, mask_int)
        if verbose:
            print(f"[FP-seeded] it={it}  ||x-z||_1(int)={L1:.6g}")

        # (e) Actualiza mejor distancia si mejora
        if L1 + 1e-12 < best_L1:
            best_L1 = L1
            no_improve = 0
        else:
            no_improve += 1

        # === 4) Chequeo de factibilidad entera ===
        # Si todas las enteras de x_rel son (casi) enteras, prueba factibilidad total
        if all(
            (not mask_int[j]) or abs(x_rel[j] - round(x_rel[j])) <= 1e-8
            for j in range(len(x_rel))
        ):
            ok, sol = check_feasible_by_fixing_integers(
                model,
                xvars,
                [
                    round(x_rel[j]) if mask_int[j] else x_rel[j]
                    for j in range(len(x_rel))
                ],
            )
            # Si se logra factibilidad → devuelve solución
            if ok:
                # Calcula valor objetivo si el modelo tiene vector de costos _v
                obj_val = (
                    sum(model._v[i] * sol[f"x[{i}]"] for i in range(len(model._v)))
                    if hasattr(model, "_v")
                    else None
                )

                if verbose:
                    print(f"[FP-seeded] solución factible en it={it}.")
                    if obj_val is not None:
                        print(
                            f"[FP-seeded] valor objetivo de la solución propuesta: {obj_val:.3f}"
                        )
                    # Si hay incumbente en el modelo, compara objetivos
                    if model.SolCount > 0:
                        best = model.ObjVal
                        print(f"[FP-seeded] incumbente actual: {best:.3f}")
                        print(
                            f"[FP-seeded] {'MEJORA!' if obj_val > best else 'no mejora.'}"
                        )
                # Devuelve el diccionario con valores de la solución
                return sol

        # === 5) Anticiclo o perturbación ===
        # Si repetimos el mismo z o no mejoramos durante varias iteraciones,
        # se introduce una perturbación aleatoria
        z_key = tuple(int(z_new[j]) if mask_int[j] else 0 for j in range(len(z_new)))
        if (z_key in z_history) or (no_improve >= 2):
            # Ordena las variables por fraccionalidad (mayor primero)
            fracs = sorted(
                [
                    (j, abs(x_rel[j] - round(x_rel[j])))
                    for j in range(len(x_rel))
                    if mask_int[j]
                ],
                key=lambda t: t[1],
                reverse=True,
            )
            frac_order = [j for j, _ in fracs]
            # Decide cuántas variables “voltear” para romper el ciclo
            # flips = 2 if len(frac_order) >= 2 else 1
            flips = max(random.randint(2, 4), int(0.01 * len(frac_order))) 
            if verbose:
                print(f"[FP-seeded] perturbación (flips={flips}) en it={it}.")
            # Aplica la perturbación en z_new
            z_new = perturb_z(
                z_new,
                vtypes,
                num_flips=flips,
                frac_order=frac_order,
                seed=seed + 1234 + it,
            )
            # Reinicia el contador de no mejora
            no_improve = 0

        # (f) Registra la configuración actual
        z_history.add(z_key)
        # (g) Actualiza z y RHS de las restricciones para la siguiente iteración
        z = z_new
        update_z_rhs(lp, cons1, cons2, z)

    # === 6) Si se agota el tiempo o las iteraciones ===
    if verbose:
        print("[FP-seeded] sin solución factible en límites dados.")
    return None
    
def make_feaspump_rounds_callback(model, xvars, collect_every_k_nodes=1, pool_cap=5000):
    """
    - Recolecta soluciones LP del nodo en model._fp_pool (saco).
    - Inyecta soluciones enteras en cola model._fp_inject (deque) con cbSetSolution().
    - Guarda historial de LB/UB/nodes si están los arrays en el modelo.
    """
    model.update()
    if not hasattr(model, "_fp_pool"):
        model._fp_pool = []
    if not hasattr(model, "_fp_inject"):
        model._fp_inject = collections.deque()
    if not hasattr(model, "_nodes_hist"):
        model._nodes_hist, model._lb_hist, model._ub_hist = [], [], []

    model._cb_calls = 0

    def cb(cb_model, where):
        # Métricas globales
        if where == GRB.Callback.MIP:
            try:
                nodes = cb_model.cbGet(GRB.Callback.MIP_NODCNT)
                bestbd = cb_model.cbGet(GRB.Callback.MIP_OBJBND)
                bestst = cb_model.cbGet(GRB.Callback.MIP_OBJBST)
                sense = model.ModelSense
                LB, UB = (bestst, bestbd) if sense == -1 else (bestbd, bestst)
                model._nodes_hist.append(nodes)
                model._lb_hist.append(LB)
                model._ub_hist.append(UB)
            except gp.GurobiError:
                pass
            return

        if where != GRB.Callback.MIPNODE:
            return

        status = cb_model.cbGet(GRB.Callback.MIPNODE_STATUS)
        if status != GRB.OPTIMAL:
            return

        # Inyectar solución entera si hay en cola
        if model._fp_inject:
            try:
                sol = model._fp_inject[0]  # peek
                cb_model.cbSetSolution(xvars, sol)
                model._fp_inject.popleft()  # consumimos una
            except gp.GurobiError:
                pass  # si Gurobi no la acepta, seguimos

        # Recolectar relajaciones
        model._cb_calls += 1
        if collect_every_k_nodes > 1 and (model._cb_calls % collect_every_k_nodes != 0):
            return
        if len(model._fp_pool) >= pool_cap:
            return

        try:
            x_rel = cb_model.cbGetNodeRel(xvars)
            model._fp_pool.append(list(x_rel))
        except gp.GurobiError:
            return

    return cb

def run_bnb_with_fp_rounds(
    model,
    xvars,
    rounds=5,  # Nº de rondas de B&B
    nodes_per_round=5000,  # Límite de nodos por ronda
    collect_every_k_nodes=1,  # Frecuencia de muestreo de relajaciones en el callback
    pool_cap=5000,  # Máximo de relajaciones a guardar
    seeds_per_round=3,  # Nº de semillas FP a probar por ronda
    fp_max_iters=40,  # Iteraciones máximas del FP
    fp_time_limit=10.0,  # Tiempo máximo del FP
    inject_as_start=True,  # Si True, setea v.start con la solución FP
    fp_verbose=False,  # Verbosidad interna del FP
    driver_verbose=True,  # Verbosidad de este driver
    presolve_fp=True,  # Ejecutar FP antes de B&B usando la raíz
    presolve_seeds=3,  # Nº de semillas FP en presolve
    presolve_try_round_pert=True,  # Generar semillas perturbadas en presolve
    presolve_time_limit=None,  # Límite de tiempo del FP en presolve (si None usa fp_time_limit)
):
    """
    Orquesta B&B intercalando Feasibility Pump (FP):
    - (Opcional) Presolve: FP seed-eado desde la relajación LP de la raíz.
    - B&B por rondas con NodeLimit; entre rondas ejecuta FP con semillas del pool
      recolectado por el callback.
    - Inyecta soluciones enteras (cbSetSolution) y/o como MIP start (v.start).
    """

    # Construye el callback que:
    #  - guarda relajaciones (en model._fp_pool)
    #  - consume/inyecta soluciones enteras (en model._fp_inject)
    cb = make_feaspump_rounds_callback(
        model, xvars, collect_every_k_nodes=collect_every_k_nodes, pool_cap=pool_cap
    )

    # Asegura que existan las estructuras que usa el callback
    if not hasattr(model, "_fp_inject"):
        model._fp_inject = []  # cola de soluciones enteras listas para inyectar
    if not hasattr(model, "_fp_pool"):
        model._fp_pool = []  # pool (saco) de relajaciones LP recolectadas

    # Guarda valores originales para restaurar al final (buen ciudadano)
    orig_NodeLimit = model.Params.NodeLimit
    orig_TimeLimit = model.Params.TimeLimit

    # Precomputa tipos y máscara de variables enteras/binarias
    vtypes = [v.VType for v in xvars]
    mask_int = [t in (GRB.BINARY, GRB.INTEGER) for t in vtypes]

    # Métrica auxiliar: distancia a integridad (sólo sobre componentes enteras/binarias)
    def frac_distance(x):
        return sum(abs(x[j] - round(x[j])) for j in range(len(x)) if mask_int[j])

    # ================= helpers internos =================

    def _root_relax_and_seeds():
        """Resuelve la relajación LP de la raíz y arma semillas para FP."""
        # Clona el modelo como LP (todas las variables continuas)
        lp_relax, var_map = clone_as_lp(model, xvars)
        # Variables x del LP clonado en el mismo orden que xvars
        x_lp_vars = [var_map[v] for v in xvars]
        # Resuelve la relajación
        lp_relax.optimize()
        # Si LP no es solucionable, no hay semillas
        if lp_relax.status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            return []

        # x0: solución de la relajación raíz
        x0 = [v.X for v in x_lp_vars]

        # Inicializa lista de semillas con la solución raíz
        seeds = [x0]

        # Si se desea, genera semillas adicionales con pequeñas perturbaciones
        if presolve_try_round_pert and presolve_seeds > 1:
            # Ordena índices por fraccionalidad (mayor primero)
            fracs = [
                (j, abs(x0[j] - round(x0[j]))) for j in range(len(x0)) if mask_int[j]
            ]
            fracs.sort(key=lambda t: t[1], reverse=True)
            # Toma un top de más fraccionales para focalizar las perturbaciones
            top_idx = [j for j, _ in fracs[: max(3, min(10, len(fracs)))]]
            rng = random.Random(12345)
            # Genera hasta presolve_seeds-1 semillas nuevas
            for s in range(1, presolve_seeds):
                xd = x0[:]  # copia de la seed base
                for j in top_idx:
                    if vtypes[j] == GRB.BINARY:
                        # Flip ocasional para explorar vecindarios
                        if rng.random() < 0.35:
                            xd[j] = 1.0 - round(x0[j])
                        else:
                            xd[j] = round(x0[j])
                    else:
                        # En enteras generales, empuja a entero cercano (aquí simple)
                        r = round(x0[j])
                        if rng.random() < 0.5:
                            xd[j] = r
                        else:
                            xd[j] = r
                seeds.append(xd)
        return seeds

    # =============== bloque principal con limpieza garantizada ===============
    try:
        # -------------------- PRESOLVE FP (opcional) --------------------
        if presolve_fp:
            if driver_verbose:
                print("\n=== Presolve FP: usando relajación LP raíz ===")

            # Asegura al menos 1 semilla
            presolve_seeds = max(1, presolve_seeds)

            # Obtiene semillas desde la relajación LP
            xvars = model.getVars()
            # num_cont = sum(1 for v in xvars if v.VType == GRB.CONTINUOUS)
            # if num_cont == 0:
            #     print("[Presolve] Modelo sin variables continuas → omitiendo relajación LP.")
            #     root_seeds = [[0.0 for _ in xvars]]  # o una inicialización aleatoria
            # else:
            root_seeds = _root_relax_and_seeds()

            best_sol_vec = None  # almacenará la primera solución FP factible

            # Itera sobre las primeras 'presolve_seeds' semillas disponibles
            for k, x0 in enumerate(root_seeds[:presolve_seeds], 1):
                if driver_verbose:
                    print(
                        f"[Presolve] FP seed {k}/{min(presolve_seeds,len(root_seeds))}  "
                        f"||frac||={frac_distance(x0):.3f}"
                    )

                # Ejecuta FP con esa semilla
                sol_dict = feasibility_pump_seeded(
                    model,
                    xvars,
                    x0,
                    max_iters=fp_max_iters,
                    time_limit=(
                        presolve_time_limit
                        if presolve_time_limit is not None
                        else fp_time_limit
                    ),
                    int_eps=1e-6,
                    seed=17 + 1000 * k,  # semilla distinta por intento
                    verbose=fp_verbose,
                )

                # Si FP entrega solución factible, la guardamos y salimos
                if sol_dict is not None:
                    best_sol_vec = [sol_dict[v.VarName] for v in xvars]
                    break

            # Si hubo solución factible en presolve, se inyecta
            if best_sol_vec is not None:
                model._fp_inject.append(best_sol_vec)  # para cbSetSolution en nodo
                if inject_as_start:
                    # Opcional: setear MIP start (v.start)
                    for v, val in zip(xvars, best_sol_vec):
                        v.start = val
                if driver_verbose:
                    print(
                        "[Presolve] FP encontró solución factible -> inyectada como MIP start."
                    )
                    print(f"✅ Solución factible encontrada en ronda {r}. Deteniendo B&B para RINS.")
                return model
            else:
                if driver_verbose:
                    print("[Presolve] FP no encontró solución factible.")

        # -------------------- RONDAS DE B&B --------------------
        for r in range(1, rounds + 1):
            if driver_verbose:
                print(f"\n=== B&B Ronda {r}/{rounds}: NodeLimit={nodes_per_round} ===")

            # Limpia el pool de relajaciones para esta ronda
            model._fp_pool = []

            # Fija el límite de nodos y ejecuta B&B con el callback
            model.Params.NodeLimit = nodes_per_round
            model.optimize(cb)

            # Recupera el pool recolectado por el callback
            pool = model._fp_pool
            if driver_verbose:
                print(f"[Ronda {r}] relajaciones recolectadas: {len(pool)}")

            # Si no hay semillas, pasa a la siguiente ronda
            if not pool:
                continue

            # Selecciona las mejores 'seeds_per_round' según cercanía a integridad
            seeds = sorted(pool, key=frac_distance)[: max(1, seeds_per_round)]

            best_sol_vec = (
                None  # reinicia “mejor” solución factible de FP en esta ronda
            )

            # Prueba FP sobre cada semilla seleccionada (primer factible gana)
            for k, x0 in enumerate(seeds, 1):
                if driver_verbose:
                    print(
                        f"[Ronda {r}] FP seed {k}/{len(seeds)}  ||frac||={frac_distance(x0):.3f}"
                    )

                sol_dict = feasibility_pump_seeded(
                    model,
                    xvars,
                    x0,
                    max_iters=fp_max_iters,
                    time_limit=fp_time_limit,
                    int_eps=1e-6,
                    seed=42 + 100 * r + k,  # semilla distinta por ronda/seed
                    verbose=fp_verbose,
                )

                if sol_dict is not None:
                    best_sol_vec = [sol_dict[v.VarName] for v in xvars]
                    break  # nos quedamos con la primera factible (puedes quitar el break si quieres)

            # Si FP encuentra solución, la encolamos para inyección y opcionalmente MIP start
            if best_sol_vec is not None:
                model._fp_inject.append(best_sol_vec)
                if inject_as_start:
                    for v, val in zip(xvars, best_sol_vec):
                        v.start = val
                if driver_verbose:
                    print(
                        f"[Ronda {r}] FP encontró solución factible -> encolada para inyección."
                    )
                    print("✅ Solución encontrada en presolve FP. Deteniendo B&B para ejecutar RINS.")
                return model

            else:
                if driver_verbose:
                    print(f"[Ronda {r}] FP no encontró solución factible.")

        # -------------------- RONDA FINAL SIN LÍMITE --------------------
        if driver_verbose:
            print("\n=== Ronda final: sin NodeLimit ===")
        model.Params.NodeLimit = GRB.INFINITY  # remueve límite de nodos
        model.optimize(cb)  # corrida final completa

    finally:
        # Siempre restaura parámetros del modelo, incluso si hubo excepciones
        if orig_NodeLimit is not None:
            model.Params.NodeLimit = orig_NodeLimit
        if orig_TimeLimit is not None:
            model.Params.TimeLimit = orig_TimeLimit

    # Devuelve el modelo optimizado
    return model