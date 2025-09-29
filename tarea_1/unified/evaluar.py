import os, sys, csv, math, heapq, json

INF = 10**18


# --------- Grafo ----------
def load_graph(path):
    # grafo.csv: i,j,c (dirigido), sin header
    edges = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            i, j, c = line.replace(",", " ").split()
            edges.append((int(i), int(j), float(c)))
    if not edges:
        raise ValueError("grafo.csv vacío")
    N = 1 + max(max(i, j) for i, j, _ in edges)
    adj = [[] for _ in range(N)]
    for i, j, c in edges:
        adj[i].append((j, c))
    return N, adj


# --------- Ruta ----------
def parse_route_txt(path):
    txt = open(path).read().strip()
    for ch in ",;\n\t":
        txt = txt.replace(ch, " ")
    toks = [t for t in txt.split(" ") if t.strip() != ""]
    return [int(t) for t in toks]


# --------- Dijkstra ----------
def dijkstra_multi_source(N, adj, sources):
    dist = [INF] * N
    pq = []
    seen = set()
    for s in sources:
        if 0 <= s < N and s not in seen:
            seen.add(s)
            dist[s] = 0.0
            heapq.heappush(pq, (0.0, s))
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


# --------- Lectura instancia ----------
def load_workers(path):
    workers = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v, r = line.replace(",", " ").split()
            workers.append((int(v), float(r)))
    return workers


# --------- Evaluación de una instancia ----------
def eval_instance(dirpath, idx):
    gpath = os.path.join(dirpath, "grafo.csv")
    ipath = os.path.join(dirpath, f"instancia{idx}.csv")
    spath = os.path.join(dirpath, f"solucion{idx}.txt")

    if not os.path.exists(gpath):
        raise FileNotFoundError(f"Falta {gpath}")
    if not os.path.exists(ipath):
        raise FileNotFoundError(f"Falta {ipath}")
    if not os.path.exists(spath):
        raise FileNotFoundError(f"Falta {spath}")

    N, adj = load_graph(gpath)
    workers = load_workers(ipath)
    W = len(workers)

    # Parsear ruta
    route = parse_route_txt(spath)

    # Esquema de salida consistente
    def pack(feasible, cost, uncovered, **extra):
        out = dict(
            feasible=bool(feasible),
            cost=float(cost),
            workers=W,
            uncovered=int(uncovered),
        )
        out.update(extra)
        return out

    # Ruta mínima razonable
    if len(route) < 2:
        return pack(False, float("inf"), W, reason="Ruta vacía o trivial")

    # Validar arcos y costear
    cost = 0.0
    for a, b in zip(route[:-1], route[1:]):
        # fuera de rango
        if not (0 <= a < N and 0 <= b < N):
            return pack(
                False, float("inf"), W, reason=f"Nodo fuera de rango en arco {a}->{b}"
            )
        # arco inexistente
        w = next((ww for nb, ww in adj[a] if nb == b), None)
        if w is None:
            return pack(False, float("inf"), W, reason=f"Arco inexistente {a}->{b}")
        cost += w

    # Cobertura
    route_nodes = list(dict.fromkeys(route))
    # Utilizamos el djikstra para ver si los trabajadores están cubiertos
    dist_to_route = dijkstra_multi_source(N, adj, route_nodes)

    uncovered_list = []
    for k, (v, r) in enumerate(workers):
        d = dist_to_route[v]
        # Si los trabajadores no están cubiertos!
        if d > r + 1e-9:
            uncovered_list.append((k, v, r, d))

    feasible = len(uncovered_list) == 0
    extra = {}
    if not feasible:
        extra["uncovered_examples"] = uncovered_list[:5]
    if route[0] != 0 or route[-1] != 0:
        extra["warning"] = "Se recomienda ruta 0→…→0; esta no inicia/termina en 0."

    return pack(feasible, cost, len(uncovered_list), **extra)


def main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Evalúa soluciones solucionX.txt contra grafo.csv e instanciaX.csv"
    )
    ap.add_argument("--dir", required=True, help="Directorio con archivos")
    ap.add_argument(
        "--idx",
        type=int,
        default=None,
        help="Índice específico (ej: 3). Si no, evalúa todos los solucion*.txt",
    )
    args = ap.parse_args()

    if args.idx is not None:
        res = eval_instance(args.dir, args.idx)
        print(json.dumps(res, indent=2))
        return

    # Buscar todos los solucion*.txt
    xs = []
    for name in os.listdir(args.dir):
        if name.startswith("solucion") and name.endswith(".txt"):
            core = name[len("solucion") : -len(".txt")]
            # acepta sufijos ("_greedy") pero solo toma enteros puros
            if core.isdigit():
                xs.append(int(core))
    xs = sorted(xs)

    agg = []
    for i in xs:
        res = eval_instance(args.dir, i)
        print("-" * 30)
        print(
            f"Instancia {i}\n- Factibilidad = {res['feasible']}\n- Costo = {res['cost']:.3f}\n"
            f"- Trabajadores = {res['workers']}\n- Sin cubrir = {res['uncovered']}"
            + (
                f"\n- Razón = {res['reason']}"
                if (not res["feasible"] and "reason" in res)
                else ""
            )
        )
        print("-" * 30)
        agg.append(res)

    ok = sum(1 for r in agg if r["feasible"])
    print(f"\nResumen: {ok}/{len(agg)} factibles.")
    if ok > 0:
        avg = sum(r["cost"] for r in agg if r["feasible"]) / ok
        print(f"Costo promedio (solo factibles): {avg:.3f}")


if __name__ == "__main__":
    main()
