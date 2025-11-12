import gurobipy as gp
from gurobipy import GRB
import time
import sys, time
from feasibilty_pump import feasibility_pump_seeded
from rins import rins_driver
from FP import run_bnb_with_fp_rounds

def crearModelo(m, TimeLimit=300): # 300 segundos = 5 minutos
    params = {
        # === DESACTIVAR COSAS ===
        "Heuristics": 0.0,  # 0 desactiva heurísticas internas
        "Cuts": 0,  # 0 desactiva cortes
        "Presolve": 0,  # 0 desactiva presolve (2 agresivo, -1 auto)
        "Symmetry": 0,  # 0 desactiva detección de simetría
        "ConcurrentMIP": 1,  # 1 desactiva concurrente (corre estrategias distintas en paralelo)
        # === FOCO DE BÚSQUEDA ===
        "MIPFocus": 0,  # 0 auto, 1 incumbente rápido, 2 gap, 3 bound
        "VarBranch": 0,  # 0 auto, 1 max infeas, 2 pseudo cost; como se elige la variable actual para hacer branching
        "NodeMethod": 1,  # 1 dual simplex en nodos (0 auto, 2 barrier); como se resuelve el LP en los nodos
        "BranchDir": 0,  # 0 auto, 1 up, -1 down
        # === TOLERANCIAS ===
        "MIPGap": 0.0,  # gap objetivo relativo deseado (p.ej. 0.01 = 1%)
        "FeasibilityTol": 1e-4,  # tolerancia de viabilidad
        "IntFeasTol": 1e-5,  # tolerancia de integralidad
        "NumericFocus": 0,  # 0-3 (3 = más robusto numéricamente)
        # === LÍMITES Y LOG ===
        "TimeLimit": TimeLimit,  # seg. (0 = sin límite) - 300 segundos = 5 minutos
        "BestObjStop": None,  # para minimización: detiene al llegar a obj <= valor
        "BestBdStop": None,  # detiene si bound <= valor
        "Threads": 0,  # 0 = auto
        "Seed": 42,
        "LogToConsole": 1,  # 1 muestra log, 0 oculta
    }

    for k, v in params.items():
        if v is not None:
            m.setParam(k, v)
    return m

TIEMPO_TOTAL = 300  
T_FP         = 240      
T_RINS       = TIEMPO_TOTAL - T_FP  

def write_solution_dict(sol_dict, out_path):
    with open(out_path, "w") as f:
        for name, val in sol_dict.items():
            f.write(f"{name} {float(val)}\n")

def main():
    t_inicio = time.time()

    inst_path = sys.argv[1]
    m = gp.read(inst_path)
    m.Params.OutputFlag = 0
    # obtenemos variables
    xvars = m.getVars()
    # # resolver relajacion 
    # m_relax = m.copy()
    # for v in m_relax.getVars():
    #     if v.VType != GRB.CONTINUOUS:
    #         v.VType = GRB.CONTINUOUS
    # m_relax.optimize()

    # # Si no se encuentra soln optima, llenar con ceros
    # if m_relax.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
    #     x0 = [0.0]*len(xvars)
    # else:
    #     var_to_relax = {v.VarName: v for v in m_relax.getVars()}
    #     x0 = [var_to_relax[v.VarName].X for v in xvars]

    # # Fase de construcción -> Feasbility Pump del aux8 
    # tiempo_usado = time.time() - t_inicio
    # T_FP = TIEMPO_TOTAL - tiempo_usado
    # sol = feasibility_pump_seeded(
    #     model=m,
    #     xvars=xvars,
    #     x0=x0,
    #     max_iters=40,
    #     time_limit=T_FP,
    #     seed=12345,
    #     verbose=False,
    # )

    # if sol is not None:
    #     print("Factible encontrada")
    #     out_name = f"resultado_{inst_path.split('/')[-1].replace('.mps.gz','')}.txt"
    #     write_solution_dict(sol, out_name)
    #     print(out_name)
    # else:
    #     print("Factible NO encontrada")
    #     out_name = f"resultado_{inst_path.split('/')[-1].replace('.mps.gz','')}.txt"
    #     # por mientras escribir 0s
    #     write_solution_dict({v.VarName: 0.0 for v in xvars}, out_name)
    #     print(out_name)

    tiempo = TIEMPO_TOTAL-(time.time()-t_inicio)
    print(tiempo)
    m = crearModelo(m, TimeLimit=tiempo)
    m.Params.FeasibilityTol = 1e-4

    tiempo = TIEMPO_TOTAL-(time.time()-t_inicio)
    # FASE FP
    model = run_bnb_with_fp_rounds(m, xvars, total_time_limit=tiempo) 
    model.Params.FeasibilityTol = 1e-4
    
    if model.SolCount > 0:
        # Extraer vector de spolución factible del modelo FP
        best_sol_vec = [v.X for v in model.getVars()]

        if not hasattr(m, "_fp_inject"):
            m._fp_inject = []
        m._fp_inject.append(best_sol_vec)

        for v, val in zip(xvars, best_sol_vec):
            v.Start = val

        print("Solución factible encontrada e inyectada como MIP start")
    else:
        print("No se encontró solución factible con FP")

    m.update()

    tiempo_usado = time.time() - t_inicio
    #restante_para_rins = max(0.0, T_RINS - max(0.0, tiempo_usado - T_FP))
    restante_para_rins = TIEMPO_TOTAL - tiempo_usado
    print(f"Tiempo restante para RINS: {restante_para_rins}")
    #m.setParam("TimeLimit", restante_para_rins) 
    rins_driver(m, xvars, rounds=3, timelimit_sub=10, node_period=200, total_time_limit=restante_para_rins)


if __name__ == "__main__":
    main()

