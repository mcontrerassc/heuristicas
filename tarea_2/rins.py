import gurobipy as gp
from gurobipy import GRB
import time
import random
random.seed(42)

# ---------------------------- RINS ----------------------------

def _grab_current_incumbent(model, xvars):
    """
    Devuelve el incumbente actual del modelo (si lo hay) como lista de valores
    en el mismo orden que xvars. Si no hay solución, devuelve None.
    """
    if model.SolCount and model.Status not in (GRB.INFEASIBLE,):
        try:
            return [v.X for v in xvars]  # leemos el valor actual de cada var
        except gp.GurobiError:
            return None
    return None

def make_rins_callback(model, xvars, node_period=200, tol_agreement=1e-6, verbose=True):
    """
    Callback tipo RINS:
      - En MIPSOL: guarda el incumbente x_inc (lista de valores).
      - En MIPNODE: cada 'node_period' nodos, si hay incumbente y el LP del nodo está óptimo,
                    captura x_lp (sol cte del LP) y encola el par (x_inc, x_lp).
      - También: si existe model._rins_candidate, intenta inyectarla vía cbSetSolution.
    """
    # cola de (x_inc, x_lp) a procesar fuera del callback
    model._rins_queue = []
    # último incumbente visto
    model._best_incumbent = None
    # simple contador de nodos para espaciar la captura
    model._node_tick = 0

    def cb(cb_model, where):
        try:
            if where == GRB.Callback.MIPSOL:
                # Nuevo incumbente: lo guardamos para luego armar vecindades RINS
                x_inc = cb_model.cbGetSolution(xvars)
                model._best_incumbent = x_inc
                if verbose:
                    obj = cb_model.cbGet(GRB.Callback.MIPSOL_OBJ)
                    print(f"[RINS:MIPSOL] Nuevo incumbente. Obj={obj:.6g}")

            elif where == GRB.Callback.MIPNODE:
                # Tenemos soluciones fraccionales
                # Si hay candidata externa guardada, intentamos inyectarla acá (rápido)
                # Acá guardamos la solución para ver donde coincide la solución entera incumbente y la fraccional actual
                cand = getattr(model, "_rins_candidate", None)
                if cand is not None:
                    try:
                        cb_model.cbSetSolution(xvars, cand)
                        if verbose:
                            print(
                                "[RINS:MIPNODE] Candidata RINS inyectada via cbSetSolution."
                            )
                    except gp.GurobiError:
                        pass
                    # limpiamos para no reinyectar cada nodo
                    model._rins_candidate = None

                # Espaciamos la captura: no queremos encolar pares en TODOS los nodos
                model._node_tick += 1
                if model._node_tick % max(1, node_period) != 0:
                    return

                # Debe existir incumbente para definir la vecindad
                if model._best_incumbent is None:
                    return

                # Nos aseguramos que el LP del nodo esté resuelto
                status = cb_model.cbGet(GRB.Callback.MIPNODE_STATUS)
                if status != GRB.OPTIMAL:
                    return

                # Extraemos la solución del LP del nodo (vector fraccional)
                x_lp = cb_model.cbGetNodeRel(xvars)

                # Encolamos una copia del par (inc, LP) para procesar FUERA del callback
                model._rins_queue.append((model._best_incumbent[:], x_lp[:]))
                if verbose:
                    nc = cb_model.cbGet(GRB.Callback.MIPNODE_NODCNT)
                    print(
                        f"[RINS:MIPNODE] Encolado par (inc, LP) en nodo {int(nc)}. Cola={len(model._rins_queue)}"
                    )

        except gp.GurobiError:
            # Si algo falla dentro del callback, no matar la corrida
            return

    return cb

def run_rins_subproblem(
    main_model, x_inc, x_lp, tol=1e-6, timelimit=5, threads=1, 
    max_vars_libres=1e9, name="rins", verbose=True
):
    """
    Arma y resuelve el sub-MIP de RINS:
      - Crea copia del modelo principal (main_model.copy()).
      - Mapea las variables de la copia en el MISMO orden que main_model._x.
      - Fija (bound-fix) las variables donde incumbente y LP 'coinciden' (según tolerancia).
      - Resuelve con un TimeLimit corto (heurística).
      - Devuelve candidata (lista de valores) o None si no halló nada.

    Regla de fijación:
      - Binarias: fijar si |x_lp - x_inc| <= tol  (x_inc es 0/1)
      - Enteras: fijar si LP está casi entero y coincide con x_inc
      - Continuas: no fijar
    """
    # Copia del modelo (evita bloquear el solve principal con un re-optimize largo)
    sub = main_model.copy()
    sub_x = [sub.getVarByName(v.VarName) for v in main_model.getVars()]
    vars_enteras_sub = [v for v in sub_x if v.VType != GRB.CONTINUOUS]

    # Contador de cuántas fijamos (solo por diagnósticos)
    nfix = 0
    vars_fijas_rins = []
    for vs, xi, xl in zip(sub_x, x_inc, x_lp):
        vtype = vs.VType

        if vtype == GRB.BINARY:
            # Si LP concuerda con el valor entero del incumbente, fijamos
            if abs(xl - (1.0 if xi >= 0.5 else 0.0)) <= tol:
                val = 1.0 if xi >= 0.5 else 0.0
                vs.LB = val
                vs.UB = val
                nfix += 1

                if isinstance(vs, gp.Var) and vs.VType != GRB.CONTINUOUS:
                    vars_fijas_rins.append(vs)

        elif vtype == GRB.INTEGER:
            # Para enteras: exigimos que LP esté casi entero y que coincida con el incumbente
            # para ver que es "casi entero" utilizamos la tolerancia
            rlp = round(xl)
            if (abs(xl - rlp) <= tol) and (abs(xi - rlp) <= tol):
                val = float(rlp)
                vs.LB = val
                vs.UB = val
                nfix += 1

                if isinstance(vs, gp.Var) and vs.VType != GRB.CONTINUOUS:
                    vars_fijas_rins.append(vs)

        else:
            # no fijamos
            pass

    # vars_libres_actuales = [v for v in vars_enteras_sub if v not in vars_fijas_rins]
    # num_libres = len(vars_libres_actuales)

    set_fijas = {v.VarName for v in vars_fijas_rins}
    vars_libres_actuales = [v for v in vars_enteras_sub if v.VarName not in set_fijas]
    num_libres = len(vars_libres_actuales)
    
    if num_libres > max_vars_libres:
        frac_dist = {v: min(v.X, 1 - v.X) for v in sub.getVars() if v not in vars_fijas_rins}
    
        # ORDENAR por fraccionarias!
        #vars_ordenadas = sorted(frac_dist.items(), key=lambda kv: kv[1])

        # Randomizar cuales se escogen
        vars_ordenadas = list(frac_dist.items())
        random.shuffle(vars_ordenadas)
        exceso = num_libres - max_vars_libres
        vars_a_fijar = [v for v, _ in vars_ordenadas[:exceso]]
        
        for v in vars_a_fijar:
            valor_fijo = round(v.X)
            v.lb = v.ub = valor_fijo
            vars_fijas_rins.add(v.VarName)
        
        if verbose:
            print(f"[RINS] Se fijaron {exceso} variables adicionales para cumplir el límite de {max_vars_libres}.")
            prom_dist = sum(frac_dist[v] for v in vars_a_fijar) / len(vars_a_fijar)
            print(f"[RINS] Promedio de infactibilidad integral de las fijadas: {prom_dist:.4f}")
            
    # Parametrizamos el sub-MIP: TimeLimit pequeño + pocos hilos → heurística “barata”
    sub.setParam("TimeLimit", timelimit)
    sub.setParam("Threads", threads)
    sub.setParam("LogToConsole", 0)  # silencioso; cambia a 1 si necesitas ver el log

    if verbose:
        print(f"[RINS] Sub-MIP: fijadas {nfix} vars. TL={timelimit}s")

    # Resolvemos el subproblema, este es el SUBMIP
    sub.optimize()

    # Si hay solución (óptimo o por corte de tiempo con solución), la empaquetamos como candidata
    cand = None
    if sub.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and sub.SolCount > 0:
        cand = [vx.X for vx in sub_x]
        if verbose:
            print(f"[RINS] Candidata obtenida. Obj(sub)={sub.ObjVal:.6g}")
    else:
        if verbose:
            print(f"[RINS] Sin candidata (Status={sub.Status}).")

    # Siempre liberar la copia (buena práctica)
    try:
        sub.dispose()
    except:
        pass

    return cand

def rins_driver(
    model,
    xvars,
    node_period=200,  # cada cuántos nodos encolar (inc,LP) desde el callback
    tol_agreement=1e-6,  # tolerancia para decidir “coinciden”
    timelimit_sub=5,  # TimeLimit del sub-MIP RINS
    rounds=2,  # cuántas rondas de “optimize principal → procesar cola → optimize corto”
    threads_sub=1,  # hilos del sub-MIP RINS (mantener bajo)
    process_even_if_optimal=True,  # seguir procesando RINS aunque el principal esté óptimo
    max_vars_libres=1e9,
    verbose=True,
    total_time_limit=None
):
    """
    Flujo:
      1) Lanza optimize() del modelo con el callback RINS (que encola pares (inc,LP)).
      2) Procesa la cola _rins_queue: para cada par, corre run_rins_subproblem(...).
      3) Si encuentra candidatas, las setea como MIP start + _rins_candidate, y lanza un optimize() corto
         para intentar inyectarlas (cbSetSolution).
    """
    t_start = time.time()

    def remaining():
        if total_time_limit is None:
            return None
        return max(0.0, total_time_limit - (time.time() - t_start))
    # Si hay presupuesto global, setear TL inicial (por seguridad)
    rem = remaining()
    if rem is not None:
        model.setParam("TimeLimit", max(1.0, rem))
        
    # Armamos el callback con parámetros de RINS
    cb = make_rins_callback(
        model,
        xvars,
        node_period=node_period,
        tol_agreement=tol_agreement,
        verbose=verbose,
    )

    for r in range(1, rounds + 1):
        # Ccorte por tiempo antes de cada ronda 
        rem = remaining()
        if rem is not None and rem <= 1.0:
            if verbose: print("RINS: sin tiempo restante antes de optimize()")
            break

        # Limita la ronda a lo que queda (para que Gurobi no se pase)
        if rem is not None:
            model.setParam("TimeLimit", rem)
        if verbose:
            print(f"\n=== RINS Ronda {r}/{rounds}: optimize() principal ===")
        model.optimize(cb)

        status = model.Status
        if verbose:
            print(
                f"[RINS-Driver] Status principal = {status} (OPTIMAL={GRB.OPTIMAL}, INFEASIBLE={GRB.INFEASIBLE})"
            )

        # Tomamos la cola que llenó el callback y la vaciamos
        queue = list(getattr(model, "_rins_queue", []))
        model._rins_queue.clear()

        # Si la cola está vacía, podemos intentar armar un par (inc, LP_relax) “a mano”
        if not queue:
            # Si el problema ya terminó y no queremos forzar nada, salimos
            if status in (GRB.OPTIMAL, GRB.INFEASIBLE) and not process_even_if_optimal:
                if verbose:
                    print("[RINS-Driver] Nada en cola y modelo terminado. Fin.")
                break

            # Si queremos procesar igual, tratamos de usar incumbente actual + relajación global
            x_inc = _grab_current_incumbent(model, xvars)
            if x_inc is not None:
                mrel = model.relax()
                mrel.setParam("OutputFlag", 0)
                # antes de optimizar la relajación, respeta presupuesto:
                rem = remaining()
                if rem is not None and rem <= 1.0:
                    try: mrel.dispose()
                    except: pass
                    break
                if rem is not None:
                    mrel.setParam("TimeLimit", min(5.0, rem))
                mrel.optimize()
                if mrel.Status == GRB.OPTIMAL:
                    # Map por nombre para construir x_lp alineado a xvars
                    sub_map = {v.VarName: v.X for v in mrel.getVars()}
                    x_lp = [sub_map.get(v.VarName, 0.0) for v in xvars]
                    queue.append((x_inc, x_lp))
                    if verbose:
                        print(
                            "[RINS-Driver] Usando (inc, LP_relax) para forzar 1 paso RINS."
                        )
                try:
                    mrel.dispose()
                except:
                    pass
            else:
                if verbose:
                    print("[RINS-Driver] Sin incumbente; fin.")
                break

        # Bandera para saber si generamos al menos una candidata
        any_candidate = False

        # Procesamos TODOS los pares encolados (último en entrar, primero en salir)
        while queue:
            #corte por tiempo dentro de la cola
            rem = remaining()
            if rem is not None and rem <= 1.0:
                if verbose: print("RINS: sin tiempo durante procesamiento de cola")
                queue.clear()
                break
            x_inc, x_lp = queue.pop()  # LIFO
            if verbose:
                print(f"[RINS-Driver] Ejecutando sub-MIP RINS (TL={timelimit_sub}s)...")

            # Lanzamos el subproblema de RINS
            cand = run_rins_subproblem(
                model,
                x_inc,
                x_lp,
                tol=tol_agreement,
                timelimit=timelimit_sub,
                threads=threads_sub,
                name=f"rins_r{r}",
                max_vars_libres=max_vars_libres,
                verbose=verbose,
            )

            # Si no hay candidata, pasamos al siguiente par de la cola
            if cand is None:
                continue

            # Si hay candidata: (i) la cargamos como MIP start
            for v, val in zip(xvars, cand):
                v.Start = val
            # (ii) la guardamos para que el callback la intente inyectar en MIPNODE
            model._rins_candidate = cand  # AQUI GUARDAMOS CANDIDATE
            any_candidate = True
            if verbose:
                print("[RINS-Driver] Candidata lista (Start + cbSetSolution).")

        # Si hubo candidata(s), hacemos un optimize corto para darle la chance de entrar
        if any_candidate:
            rem = remaining()
            if rem is not None and rem <= 1.0:
                break
            if verbose:
                print(
                    "[RINS-Driver] optimize() corto para intentar inyectar candidata..."
                )
            old_tl = model.Params.TimeLimit
            try:
                # Corrida pequeñita para no comernos todo el presupuesto del solve principal
                model.setParam("TimeLimit", min(old_tl if old_tl > 0 else 1e9, 10))
                model.optimize(cb)
            finally:
                model.setParam("TimeLimit", old_tl)

        # Si terminó y no queremos seguir “forzando” rondas, salimos
        if status in (GRB.OPTIMAL, GRB.INFEASIBLE) and not process_even_if_optimal:
            break

    if verbose:
        print("\n=== RINS Driver finalizado ===")
