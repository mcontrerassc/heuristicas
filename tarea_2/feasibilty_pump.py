import random
import time
import gurobipy as gp
from gurobipy import GRB


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
    # Fijar enteras:
    name_to_var = {v.VarName: v for v in mm.getVars()}
    for v, val in zip(xvars, z_int):
        w = name_to_var[v.VarName]
        if w.VType in (GRB.BINARY, GRB.INTEGER):
            w.lb = w.ub = float(val)
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
        if (z_key in z_history) or (no_improve >= 3):
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
            flips = 2 if len(frac_order) >= 2 else 1
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
