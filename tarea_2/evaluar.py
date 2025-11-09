# checker.py — verifica factibilidad

import sys
from gurobipy import read, GRB, LinExpr, QuadExpr

# Tolerancias
TOL_RESTR = 1e-5
TOL_INT = 1e-4
TOL_BNDS = 1e-6


def leer_solucion_txt(path):
    """Lee <NombreVariable> <Valor> por línea y devuelve dict {nombre: valor}."""
    sol = {}
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) != 2:
                continue
            name, val = parts
            try:
                sol[name] = float(val)
            except ValueError:
                pass
    return sol


def val(sol, name):
    """Valor propuesto para variable 'name'; 0.0 por omisión."""
    return float(sol.get(name, 0.0))


def chequear_factibilidad(model, sol):
    for v in model.getVars():
        x = val(sol, v.VarName)
        lb = v.LB if v.LB is not None else -GRB.INFINITY
        ub = v.UB if v.UB is not None else GRB.INFINITY

        # Bounds
        if x < lb - TOL_BNDS or x > ub + TOL_BNDS:
            return (
                f"Infactible\nDetalle: Bounds violada en {v.VarName}: "
                f"x={x}, [lb={lb}, ub={ub}] (tol {TOL_BNDS}).",
                False,
            )

        # Integridad
        if v.VType in (GRB.BINARY, GRB.INTEGER):
            if abs(x - round(x)) > TOL_INT:
                return (
                    f"Infactible\nDetalle: Variable entera {v.VarName}={x} "
                    f"se desvía {abs(x-round(x)):.6g} del entero más cercano "
                    f"(tol {TOL_INT}).",
                    False,
                )
            if v.VType == GRB.BINARY:
                if x < -TOL_INT or x > 1 + TOL_INT:
                    return (
                        f"Infactible\nDetalle: Binaria {v.VarName}={x} fuera de {{0,1}} "
                        f"(tol {TOL_INT}).",
                        False,
                    )

    # Restricciones: LHS = sum a_ij * x_j
    for c in model.getConstrs():
        row = model.getRow(c)  # LinExpr
        lhs = 0.0
        for k in range(row.size()):
            v = row.getVar(k)
            a = row.getCoeff(k)
            lhs += a * val(sol, v.VarName)

        rhs = c.RHS
        if c.Sense == GRB.LESS_EQUAL:
            viol = max(0.0, lhs - rhs)
        elif c.Sense == GRB.GREATER_EQUAL:
            viol = max(0.0, rhs - lhs)
        else:  # GRB.EQUAL
            viol = abs(lhs - rhs)

        if viol > TOL_RESTR:
            return (
                f"Infactible\nDetalle: Restricción {c.ConstrName} violada por "
                f"{viol:.6g} (tol {TOL_RESTR}).",
                False,
            )

    return "Factible", True


def objetivo(model, sol):
    """Evalúa la función objetivo con el diccionario de solución."""
    obj = model.getObjective()

    # Lineal
    if isinstance(obj, LinExpr):
        total = obj.getConstant()
        for k in range(obj.size()):
            v = obj.getVar(k)
            a = obj.getCoeff(k)
            total += a * val(sol, v.VarName)
        return total

    # Cuadrático
    if isinstance(obj, QuadExpr):
        total = obj.getLinExpr().getConstant()
        le = obj.getLinExpr()
        for k in range(le.size()):
            v = le.getVar(k)
            a = le.getCoeff(k)
            total += a * val(sol, v.VarName)
        q = obj.getQuadExpr()
        for k in range(q.size()):
            v1 = q.getVar1(k)
            v2 = q.getVar2(k)
            qcoef = q.getCoeff(k)
            total += qcoef * val(sol, v1.VarName) * val(sol, v2.VarName)
        return total

    # Fallback genérico
    try:
        total = obj.getConstant()
    except Exception:
        total = 0.0
    try:
        for k in range(obj.size()):
            v = obj.getVar(k)
            a = obj.getCoeff(k)
            total += a * val(sol, v.VarName)
    except Exception:
        pass
    return total


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python3 checker.py <instancia.mps|.mps.gz> <resultado.txt>")
        sys.exit(1)

    mps_path = sys.argv[1]
    sol_path = sys.argv[2]

    try:
        model = read(mps_path)  # admite .mps y .mps.gz
    except Exception as e:
        print(f"Error cargando instancia: {e}")
        sys.exit(1)

    sol = leer_solucion_txt(sol_path)

    msg, ok = chequear_factibilidad(model, sol)
    print(msg)
    if ok:
        z = objetivo(model, sol)
        print(f"Funcion objetivo: {z:.6f}")
