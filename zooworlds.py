"""Configuration landscapes on real backbone topologies (Internet Topology Zoo).
A world = topology + gravity traffic matrix + 14 binary configuration features
(10 per-demand reroutes to the 2nd-shortest path, 2 link sleeps, 2 meters).
Truth U(c) = mean utility over K traffic-matrix draws; a test = utility under one
random draw plus measurement noise."""
import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
FIGDIR = _os.environ.get("FIGDIR", _os.path.join(_ROOT, "figures"))
PNGDIR = _os.path.join(FIGDIR, "png")
_os.makedirs(PNGDIR, exist_ok=True)
import numpy as np, networkx as nx, itertools, math, sys, json, os
from fluid import maxmin

ZOO = _os.path.join(_ROOT, "zoo")
TOPOS = ["Abilene", "Nsfnet", "Geant2012", "Janetbackbone", "PionierL3",
         "Renater2010", "Cesnet201006", "Garr201201"]
N_REROUTE, N_SLEEP, N_METER = 10, 2, 2
NF = N_REROUTE + N_SLEEP + N_METER
WEIGHTS = (0.55, 0.25, 0.20)
QPEN = 10.0  # ms queueing penalty per saturated link on a path


def load_topo(name):
    g = nx.Graph(nx.read_graphml(f"{ZOO}/{name}.graphml"))
    g.remove_edges_from(nx.selfloop_edges(g))
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    for u, v in g.edges:
        a, b = g.nodes[u], g.nodes[v]
        try:
            la1, lo1, la2, lo2 = map(math.radians, [float(a["Latitude"]), float(a["Longitude"]),
                                                     float(b["Latitude"]), float(b["Longitude"])])
            d = 2 * 6371 * math.asin(math.sqrt(math.sin((la2-la1)/2)**2 +
                                               math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2))
        except (KeyError, ValueError):
            d = 500.0
        g.edges[u, v]["delay"] = max(0.1, d / 200.0)  # ms, light in fibre ~200 km/ms
    return nx.convert_node_labels_to_integers(g)


class World:
    def __init__(self, topo, seed, K=32, tm_sd=0.25, n_dem=24, target_util=1.4):
        rng = np.random.default_rng(seed)
        self.name = f"{topo}#{seed}"
        g = load_topo(topo); self.g = g
        n = g.number_of_nodes()
        w = rng.lognormal(0, 1, n)
        pairs = [(i, j) for i in g.nodes for j in g.nodes if i != j]
        vol = np.array([w[i] * w[j] for i, j in pairs])
        top = np.argsort(-vol)[:n_dem * 3]
        sel = rng.choice(top, size=n_dem, replace=False)  # large but not only the largest
        self.dem = [pairs[k] for k in sel]
        d = vol[sel]
        self.dl = []
        for u, v in g.edges:
            self.dl += [(u, v), (v, u)]
        self.lidx = {l: i for i, l in enumerate(self.dl)}
        self.sp = [nx.shortest_path(g, s, t, weight="delay") for s, t in self.dem]
        # capacities: uniform, scaled so that shortest-path routing peaks at target_util
        load = np.zeros(len(self.dl))
        for p, x in zip(self.sp, d):
            for k in range(len(p) - 1):
                load[self.lidx[(p[k], p[k+1])]] += x
        self.cap = np.full(len(self.dl), load.max() / target_util)
        self.d = d
        # reroute features: the demands crossing the most loaded links, 2nd-shortest path
        crit = np.argsort([-max(load[self.lidx[(p[k], p[k+1])]] for k in range(len(p)-1)) for p in self.sp])
        self.rr = []
        for j in crit:
            alt = list(itertools.islice(nx.shortest_simple_paths(g, *self.dem[j], weight="delay"), 2))
            if len(alt) == 2:
                self.rr.append((int(j), alt[1]))
            if len(self.rr) == N_REROUTE:
                break
        assert len(self.rr) == N_REROUTE, self.name
        # sleep candidates: least-loaded links whose joint removal keeps the graph connected
        und = sorted(g.edges, key=lambda e: load[self.lidx[e]] + load[self.lidx[(e[1], e[0])]])
        self.sl = []
        for e in und:
            h = g.copy(); h.remove_edges_from(self.sl + [e])
            if nx.is_connected(h):
                self.sl.append(e)
            if len(self.sl) == N_SLEEP:
                break
        # meters: two smallest demands capped at half their volume
        self.mt = [int(j) for j in np.argsort(d)[:N_METER]]
        self.tm = rng.lognormal(-tm_sd**2/2, tm_sd, size=(K, len(d)))  # mean-one multipliers
        self.dmax = 3.0 * max(sum(g.edges[p[k], p[k+1]]["delay"] for k in range(len(p)-1)) for p in self.sp) + QPEN * 3

    def evaluate(self, bits, tm=None):
        g = self.g
        tm = self.tm if tm is None else tm
        asleep = {e for i, e in enumerate(self.sl) if bits[N_REROUTE + i]}
        asleep |= {(v, u) for (u, v) in asleep}
        h = None
        paths = list(self.sp)
        for i, (j, alt) in enumerate(self.rr):
            if bits[i]:
                paths[j] = alt
        for j, p in enumerate(paths):
            if any((p[k], p[k+1]) in asleep for k in range(len(p)-1)):
                if h is None:
                    h = g.copy(); h.remove_edges_from(list(asleep))
                paths[j] = nx.shortest_path(h, *self.dem[j], weight="delay")
        A = np.zeros((len(self.dl), len(self.d)))
        prop = np.zeros(len(self.d))
        for j, p in enumerate(paths):
            for k in range(len(p) - 1):
                A[self.lidx[(p[k], p[k+1])], j] = 1
                prop[j] += g.edges[p[k], p[k+1]]["delay"]
        dem = self.d[None, :] * tm
        fcap = dem.copy()
        for i, j in enumerate(self.mt):
            if bits[N_REROUTE + N_SLEEP + i]:
                fcap[:, j] = np.minimum(fcap[:, j], 0.5 * self.d[j])
        cap = np.tile(self.cap, (tm.shape[0], 1))
        cap[:, [self.lidx[e] for e in asleep]] = 0.0
        rate = maxmin(A, cap, fcap)
        loadl = rate @ A.T
        sat = (loadl >= cap - 1e-6) & (cap > 0)
        nsat = sat.astype(float) @ A                       # saturated links per path
        delay = prop[None, :] + QPEN * nsat
        T = np.mean(np.minimum(rate / dem, 1.0), axis=1)
        L = np.mean(np.maximum(0.0, 1.0 - delay / self.dmax), axis=1)
        E = len(asleep) / 2 / N_SLEEP
        return WEIGHTS[0] * T + WEIGHTS[1] * L + WEIGHTS[2] * E   # (K,)

    def enumerate(self):
        M = np.zeros((1 << NF, self.tm.shape[0]))
        for c in range(1 << NF):
            bits = [(c >> i) & 1 for i in range(NF)]
            M[c] = self.evaluate(bits)
        return M

    def nominal(self, est_sd=0.0, seed=0):
        """Twin view used for pre-screening: utility at a single traffic matrix, the
        nominal one (est_sd=0) or one estimated with multiplicative error est_sd."""
        V = np.zeros(1 << NF)
        rng = np.random.default_rng(10_000 + seed)
        tmv = (rng.lognormal(-est_sd**2 / 2, est_sd, size=(1, len(self.d))) if est_sd > 0
               else np.ones((1, len(self.d))))
        for c in range(1 << NF):
            V[c] = self.evaluate([(c >> i) & 1 for i in range(NF)], tm=tmv)[0]
        return V


if __name__ == "__main__":
    os.makedirs("data/zoo", exist_ok=True)
    for topo in (sys.argv[1].split(",") if len(sys.argv) > 1 else TOPOS):
        for seed in range(3):
            fn = f"data/zoo/{topo}_{seed}.npz"
            if os.path.exists(fn):
                continue
            w = World(topo, seed)
            # one batched pass: K traffic draws, the nominal matrix and the estimated matrix
            rng = np.random.default_rng(10_000 + seed)
            est = rng.lognormal(-0.3**2 / 2, 0.3, size=(1, len(w.d)))
            tms = np.vstack([w.tm, np.ones((1, len(w.d))), est])
            ALL = np.zeros((1 << NF, tms.shape[0]))
            for c in range(1 << NF):
                ALL[c] = w.evaluate([(c >> i) & 1 for i in range(NF)], tm=tms)
            K = w.tm.shape[0]; M = ALL[:, :K]; V0 = ALL[:, K]; V = ALL[:, K + 1]
            np.savez_compressed(fn, M=M, V=V, V0=V0, n=w.g.number_of_nodes(), m=w.g.number_of_edges())
            U = M.mean(1)
            print(w.name, "nodes", w.g.number_of_nodes(), "U* %.3f mean %.3f sd %.3f noise-sd %.3f corr(V0,U) %.3f corr(V,U) %.3f"
                  % (U.max(), U.mean(), U.std(), M.std(1).mean(), np.corrcoef(U, V0)[0, 1],
                     np.corrcoef(U, V)[0, 1]), flush=True)
