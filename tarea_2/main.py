import gurobipy as gp
from gurobipy import GRB
import sys, time
from feasibilty_pump import feasibility_pump_seeded, check_feasible_by_fixing_integers

# max total 5min
TMAX_TOTAL = 300
# 4 minutos para FP (?)
T_FP = 240 

def write_solution_dict(sol_dict, out_path):
    with open(out_path, "w") as f:
        for name, val in sol_dict.items():
            f.write(f"{name} {float(val)}\n")

def main():
    inst_path = sys.argv[1]
    m = gp.read(inst_path)
    m.Params.OutputFlag = 0
    # obtenemos variables
    xvars = m.getVars()
    # resolver relajacion 
    m_relax = m.copy()
    for v in m_relax.getVars():
        if v.VType != GRB.CONTINUOUS:
            v.VType = GRB.CONTINUOUS
    m_relax.optimize()

    # Si no se encuentra soln optima, llenar con ceros
    if m_relax.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        x0 = [0.0]*len(xvars)
    else:
        var_to_relax = {v.VarName: v for v in m_relax.getVars()}
        x0 = [var_to_relax[v.VarName].X for v in xvars]

    # Fase de construcción -> Feasbility Pump del aux8 
    sol = feasibility_pump_seeded(
        model=m,
        xvars=xvars,
        x0=x0,
        max_iters=40,
        time_limit=T_FP,
        seed=12345,
        verbose=False,
    )

    if sol is not None:
        print("Factible encontrada")
        out_name = f"resultado_{inst_path.split('/')[-1].replace('.mps.gz','')}.txt"
        write_solution_dict(sol, out_name)
        print(out_name)
    else:
        print("Factible NO encontrada")
        out_name = f"resultado_{inst_path.split('/')[-1].replace('.mps.gz','')}.txt"
        # por mientras escribir 0s
        write_solution_dict({v.VarName: 0.0 for v in xvars}, out_name)
        print(out_name)

if __name__ == "__main__":
    main()

