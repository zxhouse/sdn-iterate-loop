import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
FIGDIR = _os.environ.get("FIGDIR", _os.path.join(_ROOT, "figures"))
PNGDIR = _os.path.join(FIGDIR, "png")
_os.makedirs(PNGDIR, exist_ok=True)
#!/usr/bin/env python3
"""Large programmable testbed: the GEANT 2012 backbone (40 PoPs, 61 links, Internet
Topology Zoo) emulated with one Open vSwitch bridge per PoP under OpenFlow 1.3.
Twelve elastic demands, a latency probe and three uncontrolled cross-traffic sources.
Ten configuration knobs: five demand reroutes, one probe reroute, two meters and two
link-sleep groups. The controller recomputes paths around sleeping links, compiles
per-switch flow and meter tables, and pushes only the tables that changed (bundles)."""
import subprocess, json, time, os, sys, statistics, itertools, random, shlex, math
import numpy as np, networkx as nx

ZOO = _os.path.join(_ROOT, "zoo", "Geant2012.graphml")
STATE = "/tmp/bigbed_state.json"
SEED = 7
N_DEM = 12
RTT_MAX = 100.0
WEIGHTS = {"T": 0.55, "L": 0.25, "E": 0.20}


def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stderr}")
    return r.stdout


def load_graph():
    g = nx.Graph(nx.read_graphml(ZOO))
    g.remove_edges_from(nx.selfloop_edges(g))
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    g = nx.convert_node_labels_to_integers(g, label_attribute="zoo_id")
    for u, v in g.edges:
        a, b = g.nodes[u], g.nodes[v]
        try:
            la1, lo1, la2, lo2 = map(math.radians, [float(a["Latitude"]), float(a["Longitude"]),
                                                     float(b["Latitude"]), float(b["Longitude"])])
        except KeyError:
            g.edges[u, v]["delay"] = 2.5; continue
        d = 2 * 6371 * math.asin(math.sqrt(math.sin((la2 - la1) / 2) ** 2 +
                                           math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2))
        g.edges[u, v]["delay"] = max(0.1, d / 200.0)
    return g


G = load_graph()
SW = [f"g{i}" for i in G.nodes]
rng = np.random.default_rng(SEED)
# ---- demands (gravity model), capacities, knobs
w = rng.lognormal(0, 1, G.number_of_nodes())
pairs = [(i, j) for i in G.nodes for j in G.nodes if i != j]
vol = np.array([w[i] * w[j] for i, j in pairs])
top = np.argsort(-vol)[:N_DEM * 4]
sel = rng.choice(top, size=N_DEM, replace=False)
DEM = [pairs[k] for k in sel]
raw = vol[sel]
DRATE = list(np.round(0.8 + 1.7 * (raw - raw.min()) / (raw.max() - raw.min()), 1))  # Mbit/s in [0.8, 2.5]
SP = [nx.shortest_path(G, s, t, weight="delay") for s, t in DEM]
BG_PAIRS = []
load = {}
for p, d in zip(SP, DRATE):
    for k in range(len(p) - 1):
        e = tuple(sorted((p[k], p[k + 1]))); load[e] = load.get(e, 0) + d
hot = sorted(load, key=load.get, reverse=True)
# uncontrolled cross-traffic over the three most loaded links
BG = {}
for i, e in enumerate(hot[:3]):
    BG[f"B{i+1}"] = (e[0], e[1])
BG_RATES = [0.0, 1.0, 2.0, 3.0]
CAP_HOT = max(load.values()) / 1.4
CAP = {tuple(sorted(e)): (round(CAP_HOT, 1) if load.get(tuple(sorted(e)), 0) > 0.5 * max(load.values()) else round(CAP_HOT, 1))
       for e in G.edges}
# reroute knobs: five demands crossing the hottest links -> second-shortest path
crit = sorted(range(N_DEM), key=lambda j: -max(load[tuple(sorted((SP[j][k], SP[j][k + 1])))] for k in range(len(SP[j]) - 1)))
RR = []
for j in crit:
    alt = list(itertools.islice(nx.shortest_simple_paths(G, *DEM[j], weight="delay"), 2))
    if len(alt) == 2:
        RR.append((j, alt[1]))
    if len(RR) == 5:
        break
# probe between two PoPs whose shortest path crosses the hottest link
hl = hot[0]
cands = [(a, b) for a in G.nodes for b in G.nodes if a < b and a not in DEM and len(nx.shortest_path(G, a, b)) >= 4]
PROBE = None
for a, b in sorted(cands, key=lambda ab: (ab[0] * 7 + ab[1] * 13) % 97):
    p = nx.shortest_path(G, a, b, weight="delay")
    if any(tuple(sorted((p[k], p[k + 1]))) == hl for k in range(len(p) - 1)):
        alt = list(itertools.islice(nx.shortest_simple_paths(G, a, b, weight="delay"), 2))
        if len(alt) == 2:
            PROBE = (a, b, p, alt[1]); break
# sleep groups: lightly loaded links, each group keeps the graph connected jointly
und = sorted(G.edges, key=lambda e: load.get(tuple(sorted(e)), 0))
groups, used = [], []
for e in und:
    if load.get(tuple(sorted(e)), 0) > 0:
        continue
    h = G.copy(); h.remove_edges_from(used + [e])
    if nx.is_connected(h):
        used.append(e)
    if len(used) == 4:
        break
SLEEP = {"sleepA": used[:2], "sleepB": used[2:4]}
MT = sorted(range(N_DEM), key=lambda j: DRATE[j])[:2]
FEATURES = [f"D{j}alt" for j, _ in RR] + ["PRalt", f"meterD{MT[0]}", f"meterD{MT[1]}", "sleepA", "sleepB"]
NF = len(FEATURES)

# ---- naming
HOSTS = {}
for j, (s, t) in enumerate(DEM):
    HOSTS[f"hs{j}"] = s; HOSTS[f"hd{j}"] = t
HOSTS["hpa"] = PROBE[0]; HOSTS["hpb"] = PROBE[1]
for b, (u, v) in BG.items():
    HOSTS[f"h{b}s"] = u; HOSTS[f"h{b}d"] = v
HLIST = list(HOSTS)
def ip_of(h): return f"10.1.{HLIST.index(h) // 200}.{HLIST.index(h) % 200 + 1}"
def mac_of(h): i = HLIST.index(h); return "00:00:00:01:%02x:%02x" % (i // 256, i % 256)
def lname(a, b): return f"v{a}x{b}"


PORTS = {}


def build():
    teardown()
    for i in G.nodes:
        sh(f"ovs-vsctl --may-exist add-br g{i} -- set bridge g{i} datapath_type=netdev protocols=OpenFlow13 "
           f"fail_mode=secure other-config:datapath-id={i+1:016x}")
    nxt = {i: 1 for i in G.nodes}
    for (a, b) in G.edges:
        cap = CAP[tuple(sorted((a, b)))]
        sh(f"ip link add {lname(a,b)} type veth peer name {lname(b,a)}")
        for x, y in ((a, b), (b, a)):
            n = lname(x, y)
            sh(f"ip link set {n} up"); sh(f"ethtool -K {n} tx off rx off", check=False)
            sh(f"ovs-vsctl add-port g{x} {n} -- set interface {n} ofport_request={nxt[x]}")
            PORTS[f"{x}|{y}"] = nxt[x]; nxt[x] += 1
            sh(f"tc qdisc add dev {n} root tbf rate {cap}mbit burst 8kb latency 50ms")
    for h, s in HOSTS.items():
        sh(f"ip netns add {h}")
        hi, si = f"{h}-e0", f"g{s}-{h}"[:15]
        sh(f"ip link add {hi} type veth peer name {si}")
        sh(f"ip link set {hi} netns {h}")
        sh(f"ip netns exec {h} ip link set {hi} address {mac_of(h)}")
        sh(f"ip netns exec {h} ip addr add {ip_of(h)}/16 dev {hi}")
        sh(f"ip netns exec {h} ip link set {hi} up; ip netns exec {h} ip link set lo up")
        sh(f"ip netns exec {h} ethtool -K {hi} tx off rx off", check=False)
        sh(f"ip link set {si} up"); sh(f"ethtool -K {si} tx off rx off", check=False)
        sh(f"ovs-vsctl add-port g{s} {si} -- set interface {si} ofport_request={nxt[s]}")
        PORTS[f"{s}|{h}"] = nxt[s]; nxt[s] += 1
    for h in HOSTS:
        for g2 in HOSTS:
            if g2 != h:
                sh(f"ip netns exec {h} ip neigh replace {ip_of(g2)} lladdr {mac_of(g2)} dev {h}-e0 nud permanent")
    for j, d in enumerate(DRATE):
        sh(f"ip netns exec hs{j} tc qdisc add dev hs{j}-e0 root tbf rate {d}mbit burst 8kb latency 20ms")
    start_servers()
    json.dump({"ports": PORTS, "tables": {}}, open(STATE, "w"))


def start_servers():
    for j in range(N_DEM):
        sh(f"ip netns exec hd{j} iperf3 -s -D -p {5400+j}")
    for i, b in enumerate(BG):
        sh(f"ip netns exec h{b}d iperf3 -s -D -p {5500+i}")


def teardown():
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    for line in out.splitlines()[1:]:
        pid, args = line.strip().split(None, 1)
        if args.startswith("iperf3 "):
            os.kill(int(pid), 9)
    for b in sh("ovs-vsctl list-br", check=False).split():
        sh(f"ovs-vsctl --if-exists del-br {b}", check=False)
    sh("ip -all netns delete", check=False)
    for (a, b) in G.edges:
        sh(f"ip link del {lname(a,b)}", check=False)


def resolve(x):
    asleep = set()
    for k, f in enumerate(FEATURES):
        if f in SLEEP and x[k]:
            for (a, b) in SLEEP[f]:
                asleep |= {(a, b), (b, a)}
    h = G.copy(); h.remove_edges_from([e for e in asleep])
    def ok(p): return all((p[k], p[k + 1]) not in asleep for k in range(len(p) - 1))
    def fix(p, s, t): return p if ok(p) else nx.shortest_path(h, s, t, weight="delay")
    paths = {}
    rrmap = {j: alt for j, alt in RR}
    for j, (s, t) in enumerate(DEM):
        k = [i for i, (jj, _) in enumerate(RR) if jj == j]
        p = rrmap[j] if (k and x[k[0]]) else SP[j]
        paths[f"D{j}"] = fix(p, s, t)
    pp = PROBE[3] if x[FEATURES.index("PRalt")] else PROBE[2]
    paths["PR"] = fix(pp, PROBE[0], PROBE[1])
    for b, (u, v) in BG.items():
        paths[b] = fix([u, v], u, v)
    return paths, asleep


def compile_config(x, ports):
    paths, asleep = resolve(x)
    flows = {i: ["priority=0,actions=drop"] for i in G.nodes}
    meters = {i: [] for i in G.nodes}
    mon = {MT[0]: x[FEATURES.index(f"meterD{MT[0]}")], MT[1]: x[FEATURES.index(f"meterD{MT[1]}")]}
    ends = {f"D{j}": (f"hs{j}", f"hd{j}") for j in range(N_DEM)}
    ends["PR"] = ("hpa", "hpb")
    for b in BG:
        ends[b] = (f"h{b}s", f"h{b}d")
    for f, p in paths.items():
        src, dst = ends[f]
        for k, s in enumerate(p):
            of = ports[f"{s}|{p[k+1]}"] if k + 1 < len(p) else ports[f"{s}|{dst}"]
            orv = ports[f"{s}|{p[k-1]}"] if k > 0 else ports[f"{s}|{src}"]
            m = ""
            if k == 0 and f.startswith("D") and mon.get(int(f[1:]), 0):
                mid = MT.index(int(f[1:])) + 1
                m = f"meter:{mid},"
                meters[s].append(f"meter={mid},kbps,band=type=drop,rate={int(500 * DRATE[int(f[1:])])}")
            flows[s].append(f"priority=100,ip,nw_src={ip_of(src)},nw_dst={ip_of(dst)},actions={m}output:{of}")
            flows[s].append(f"priority=100,ip,nw_src={ip_of(dst)},nw_dst={ip_of(src)},actions=output:{orv}")
    return flows, meters, paths, asleep


def install(x):
    st = json.load(open(STATE))
    flows, meters, paths, asleep = compile_config(x, st["ports"])
    tables = st.get("tables", {})
    for i in G.nodes:
        key = "\n".join(sorted(flows[i])) + "#" + "\n".join(sorted(meters[i]))
        if tables.get(str(i)) == key:
            continue
        sh(f"ovs-ofctl -O OpenFlow13 del-meters g{i}", check=False)
        for m in meters[i]:
            sh(f"ovs-ofctl -O OpenFlow13 add-meter g{i} {m}")
        fn = f"/tmp/bflows_{i}.txt"
        open(fn, "w").write("\n".join(flows[i]) + "\n")
        sh(f"ovs-ofctl -O OpenFlow13 --bundle replace-flows g{i} {fn}")
        tables[str(i)] = key
    for (a, b) in G.edges:
        st_ = "down" if (a, b) in asleep else "up"
        if tables.get(f"L{a}-{b}") != st_:
            sh(f"ip link set {lname(a,b)} {st_}; ip link set {lname(b,a)} {st_}")
            tables[f"L{a}-{b}"] = st_
    st["tables"] = tables; json.dump(st, open(STATE, "w"))
    return paths, asleep


def measure(dur, bg):
    procs, bgp = {}, []
    for i, (b, r) in enumerate(bg.items()):
        if r > 0:
            cmd = f"ip netns exec h{b}s iperf3 -u -c {ip_of(f'h{b}d')} -p {5500+i} -t {int(dur)} -b {r}M -l 1200"
            bgp.append(subprocess.Popen(shlex.split(cmd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for j in range(N_DEM):
        cmd = f"ip netns exec hs{j} iperf3 -c {ip_of(f'hd{j}')} -p {5400+j} -t {int(dur)} -l 16K -J --connect-timeout 1000"
        procs[j] = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    n = max(3, int((dur + 0.5) / 0.1))
    pp = subprocess.Popen(shlex.split(f"ip netns exec hpa ping -n -q -i 0.1 -W 1 -c {n} {ip_of('hpb')}"),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    thr = {}
    for j, p in procs.items():
        try:
            out, _ = p.communicate(timeout=dur + 12)
            thr[j] = json.loads(out)["end"]["sum_received"]["bits_per_second"] / 1e6
        except Exception:
            p.kill(); thr[j] = float("nan")
    out, _ = pp.communicate(timeout=dur + 12)
    rtt, loss = RTT_MAX, 100.0
    for line in out.splitlines():
        if "rtt min/avg/max" in line:
            rtt = float(line.split("=")[1].split("/")[1])
        if "packet loss" in line:
            loss = float(line.split("%")[0].split()[-1])
    if loss >= 99:
        rtt = RTT_MAX
    for p in bgp:
        try: p.wait(timeout=dur + 12)
        except Exception: p.kill()
    return thr, rtt, loss


def utility(thr, rtt, asleep):
    T = statistics.mean(min(thr[j] / DRATE[j], 1.0) for j in range(N_DEM))
    L = max(0.0, 1.0 - rtt / RTT_MAX)
    E = (len(asleep) // 2) / 4.0
    return WEIGHTS["T"] * T + WEIGHTS["L"] * L + WEIGHTS["E"] * E, T, L, E


def test(x, dur=1, settle=0.3, rng=random, bg=None):
    if bg is None:
        bg = {b: rng.choice(BG_RATES) for b in BG}
    paths, asleep = install(x)
    time.sleep(settle)
    thr, rtt, loss = measure(dur, bg)
    bad = any(math.isnan(v) for v in thr.values())
    if bad:  # a stuck server: restart servers and repeat the test once
        teardown_servers(); start_servers(); time.sleep(0.5)
        thr, rtt, loss = measure(dur, bg)
        thr = {j: (0.0 if math.isnan(v) else v) for j, v in thr.items()}
    u, T, L, E = utility(thr, rtt, asleep)
    return {"x": "".join(map(str, x)), "u": u, "T": T, "L": L, "E": E, "rtt": rtt, "loss": loss,
            "thr": thr, "bg": bg, "dur": dur, "t": time.time(), "retried": bad}


def teardown_servers():
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    for line in out.splitlines()[1:]:
        pid, args = line.strip().split(None, 1)
        if args.startswith("iperf3 "):
            os.kill(int(pid), 9)
    time.sleep(0.5)


def describe():
    return {"switches": G.number_of_nodes(), "links": G.number_of_edges(), "hosts": len(HOSTS),
            "demands": [(int(s), int(t), float(d)) for (s, t), d in zip(DEM, DRATE)],
            "cap": float(CAP_HOT), "features": FEATURES, "rr": [(int(j), [int(v) for v in p]) for j, p in RR],
            "probe": [int(PROBE[0]), int(PROBE[1]), [int(v) for v in PROBE[2]], [int(v) for v in PROBE[3]]],
            "sleep": {k: [(int(a), int(b)) for a, b in v] for k, v in SLEEP.items()}, "meters": [int(m) for m in MT],
            "bg": {k: [int(a), int(b)] for k, (a, b) in BG.items()},
            "labels": {int(n): G.nodes[n].get("label", str(n)) for n in G.nodes}}


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "build":
        build(); print("built", len(SW), "switches", len(HOSTS), "hosts")
    elif c == "describe":
        print(json.dumps(describe(), indent=1))
    elif c == "test":
        print(json.dumps(test([int(ch) for ch in sys.argv[2]])))
    elif c == "teardown":
        teardown()
