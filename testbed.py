#!/usr/bin/env python3
"""Programmable SDN testbed: Linux network namespaces as hosts, Open vSwitch
(userspace datapath) bridges as OpenFlow 1.3 switches, HTB/TBF-shaped veth links.
A configuration (a 'prototype') is a binary vector over FEATURES; the controller
compiles it into OpenFlow flow and meter tables and installs them atomically
(ovs-ofctl --bundle). A 'test' runs paced TCP flows (iperf3) and an ICMP probe and
returns per-flow throughput, probe RTT and the utility U."""
import subprocess, json, time, os, sys, statistics, itertools, random, shlex

SW = ["s1", "s2", "s3", "s4", "s5", "s6"]
# inter-switch links and capacities (Mbit/s)
LINKS = {("s1", "s2"): 10, ("s2", "s5"): 10, ("s1", "s3"): 6, ("s3", "s5"): 6,
         ("s1", "s4"): 5, ("s4", "s6"): 5, ("s6", "s5"): 5, ("s2", "s3"): 5}
# hosts: name -> (switch, ip index)
HOSTS = {"h1": "s1", "h2": "s1", "h3": "s1", "h9": "s1", "h11": "s1", "h4": "s2",
         "h5": "s5", "h6": "s5", "h7": "s5", "h8": "s5", "h10": "s5", "h12": "s5",
         "h13": "s1", "h14": "s5", "h15": "s1", "h16": "s5"}
# elastic flows: name -> (src, dst, demand Mbit/s, port, [P0, P1])
FLOWS = {
    "F1": ("h1", "h5", 4.5, 5201, [["s1", "s2", "s5"], ["s1", "s3", "s5"]]),
    "F2": ("h2", "h6", 4, 5202, [["s1", "s2", "s5"], ["s1", "s4", "s6", "s5"]]),
    "F3": ("h3", "h7", 3, 5203, [["s1", "s2", "s5"], ["s1", "s3", "s5"]]),
    "F4": ("h4", "h8", 3.5, 5204, [["s2", "s5"], ["s2", "s3", "s5"]]),
    "F5": ("h11", "h12", 2.5, 5205, [["s1", "s3", "s5"], ["s1", "s4", "s6", "s5"]]),
}
# exogenous cross-traffic (not under the controller's choice): fixed paths, random rate per test
BG = {"B1": ("h13", "h14", 5301, ["s1", "s2", "s5"]), "B2": ("h15", "h16", 5302, ["s1", "s3", "s5"])}
BG_RATES = [0.0, 1.0, 2.0, 3.0]  # Mbit/s, drawn independently per test
PROBE = ("h9", "h10", [["s1", "s2", "s5"], ["s1", "s4", "s6", "s5"]])
FEATURES = ["F1alt", "F2alt", "F3alt", "F4alt", "F5alt", "PRalt",
            "meterF1", "meterF2", "sleepS4", "sleepS23"]
NF = len(FEATURES)
METERS = {"meterF1": ("F1", 2500), "meterF2": ("F2", 2000)}  # kbit/s
SLEEP = {"sleepS4": [("s1", "s4"), ("s4", "s6"), ("s6", "s5")], "sleepS23": [("s2", "s3")]}
WEIGHTS = {"T": 0.55, "L": 0.25, "E": 0.20}
RTT_MAX = 60.0  # ms


def sh(cmd, check=True, quiet=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd}\n{r.stderr}")
    return r.stdout


def ip_of(h): return f"10.0.0.{int(h[1:])}"
def mac_of(h): return "00:00:00:00:00:%02x" % int(h[1:])
def lname(a, b): return f"{a}{b}"  # veth end attached to switch a, towards b


PORTS = {}  # (switch, neighbour-or-host) -> ofport


def build():
    teardown(quiet=True)
    sh("mkdir -p /var/run/openvswitch")
    for s in SW:
        sh(f"ovs-vsctl --may-exist add-br {s} -- set bridge {s} datapath_type=netdev "
           f"protocols=OpenFlow13 fail_mode=secure other-config:datapath-id=000000000000000{s[1]}")
    nxt = {s: 1 for s in SW}
    for (a, b), cap in LINKS.items():
        sh(f"ip link add {lname(a,b)} type veth peer name {lname(b,a)}")
        for x, y in ((a, b), (b, a)):
            n = lname(x, y)
            sh(f"ip link set {n} up")
            sh(f"ethtool -K {n} tx off rx off", check=False)
            sh(f"ovs-vsctl add-port {x} {n} -- set interface {n} ofport_request={nxt[x]}")
            PORTS[(x, y)] = nxt[x]; nxt[x] += 1
            # shaping on egress of each veth end: queue up to 50 ms
            sh(f"tc qdisc add dev {n} root tbf rate {cap}mbit burst 16kb latency 50ms")
    for h, s in HOSTS.items():
        sh(f"ip netns add {h}")
        hi, si = f"{h}-eth0", f"{s}-{h}"
        sh(f"ip link add {hi} type veth peer name {si}")
        sh(f"ip link set {hi} netns {h}")
        sh(f"ip netns exec {h} ip link set {hi} address {mac_of(h)}")
        sh(f"ip netns exec {h} ip addr add {ip_of(h)}/24 dev {hi}")
        sh(f"ip netns exec {h} ip link set {hi} up; ip netns exec {h} ip link set lo up")
        sh(f"ip netns exec {h} ethtool -K {hi} tx off rx off", check=False)
        sh(f"ip link set {si} up"); sh(f"ethtool -K {si} tx off rx off", check=False)
        sh(f"ovs-vsctl add-port {s} {si} -- set interface {si} ofport_request={nxt[s]}")
        PORTS[(s, h)] = nxt[s]; nxt[s] += 1
    for h in HOSTS:  # static ARP: no flooding needed, the controller owns all forwarding
        for g in HOSTS:
            if g != h:
                sh(f"ip netns exec {h} ip neigh replace {ip_of(g)} lladdr {mac_of(g)} dev {h}-eth0 nud permanent")
    # access links: the demand of each elastic flow is its access rate (kernel TBF in the source namespace)
    for f, (src, dst, d, port, paths) in FLOWS.items():
        sh(f"ip netns exec {src} tc qdisc add dev {src}-eth0 root tbf rate {d}mbit burst 8kb latency 20ms")
    # persistent iperf3 servers on sinks
    for f, (src, dst, d, port, paths) in FLOWS.items():
        sh(f"ip netns exec {dst} iperf3 -s -D -p {port}")
    for b, (src, dst, port, p) in BG.items():
        sh(f"ip netns exec {dst} iperf3 -s -D -p {port}")
    json.dump({f"{a}|{b}": p for (a, b), p in PORTS.items()}, open("/tmp/sdn_ports.json", "w"))


def load_ports():
    global PORTS
    if not PORTS:
        PORTS = {tuple(k.split("|")): v for k, v in json.load(open("/tmp/sdn_ports.json")).items()}


def teardown(quiet=False):
    sh("pkill -f 'iperf3 -s'", check=False)
    for s in SW:
        sh(f"ovs-vsctl --if-exists del-br {s}", check=False)
    sh("ip -all netns delete", check=False)
    for (a, b) in LINKS:
        sh(f"ip link del {lname(a,b)}", check=False)


def resolve_paths(x):
    """Map a configuration to concrete paths, applying feature interactions:
    a path through a sleeping link falls back to the flow's default path."""
    asleep = set()
    for i, f in enumerate(FEATURES):
        if f in SLEEP and x[i]:
            for (a, b) in SLEEP[f]:
                asleep |= {(a, b), (b, a)}
    def ok(p): return all((p[k], p[k+1]) not in asleep for k in range(len(p)-1))
    paths = {}
    for i, (f, (src, dst, d, port, P)) in enumerate(FLOWS.items()):
        p = P[1] if x[i] else P[0]
        paths[f] = p if ok(p) else P[0]
    pp = PROBE[2][1] if x[5] else PROBE[2][0]
    paths["PR"] = pp if ok(pp) else PROBE[2][0]
    return paths, asleep


def compile_config(x):
    """Return {switch: [flow strings]}, {switch: [meter strings]}."""
    load_ports()
    paths, asleep = resolve_paths(x)
    flows = {s: ["priority=0,actions=drop", "priority=1,arp,actions=drop"] for s in SW}
    meters = {s: [] for s in SW}
    meter_of = {}
    for mi, (feat, (fl, rate)) in enumerate(METERS.items(), start=1):
        if x[FEATURES.index(feat)]:
            meter_of[fl] = mi
    pairs = [(f, FLOWS[f][0], FLOWS[f][1]) for f in FLOWS] + [("PR", PROBE[0], PROBE[1])]
    pairs += [(b, BG[b][0], BG[b][1]) for b in BG]
    for b in BG:
        paths[b] = BG[b][3]
    for f, src, dst in pairs:
        p = paths[f]
        for k, s in enumerate(p):
            out_fwd = PORTS[(s, p[k+1])] if k + 1 < len(p) else PORTS[(s, dst)]
            out_rev = PORTS[(s, p[k-1])] if k > 0 else PORTS[(s, src)]
            m = ""
            if k == 0 and f in meter_of:
                m = f"meter:{meter_of[f]},"
                meters[s].append(f"meter={meter_of[f]},kbps,band=type=drop,rate={METERS[[q for q in METERS if METERS[q][0]==f][0]][1]}")
            flows[s].append(f"priority=100,ip,nw_src={ip_of(src)},nw_dst={ip_of(dst)},actions={m}output:{out_fwd}")
            flows[s].append(f"priority=100,ip,nw_src={ip_of(dst)},nw_dst={ip_of(src)},actions=output:{out_rev}")
    return flows, meters, paths, asleep


def install(x):
    flows, meters, paths, asleep = compile_config(x)
    for s in SW:
        sh(f"ovs-ofctl -O OpenFlow13 del-meters {s}", check=False)
        for m in meters[s]:
            sh(f"ovs-ofctl -O OpenFlow13 add-meter {s} {m}")
        fn = f"/tmp/flows_{s}.txt"
        open(fn, "w").write("\n".join(flows[s]) + "\n")
        sh(f"ovs-ofctl -O OpenFlow13 --bundle replace-flows {s} {fn}")
    # power state of links (administratively down when asleep)
    for (a, b) in LINKS:
        st = "down" if (a, b) in asleep else "up"
        sh(f"ip link set {lname(a,b)} {st}; ip link set {lname(b,a)} {st}")
    return paths, asleep


def energy_score(asleep):
    on = sum(1 for l in LINKS if l not in asleep)
    return 1.0 - on / len(LINKS) + 0.0  # 0 when all on


def measure(dur=2.0, bg=None):
    """Run all elastic flows, the cross-traffic and the probe concurrently for dur seconds."""
    procs = {}
    bgp = []
    for b, r in (bg or {}).items():
        if r > 0:
            src, dst, port, p = BG[b]
            cmd = f"ip netns exec {src} iperf3 -u -c {ip_of(dst)} -p {port} -t {dur} -b {r}M -l 1200"
            bgp.append(subprocess.Popen(shlex.split(cmd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for f, (src, dst, d, port, P) in FLOWS.items():
        cmd = f"ip netns exec {src} iperf3 -c {ip_of(dst)} -p {port} -t {dur} -l 16K -J -O 0 --connect-timeout 1000"
        procs[f] = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    n = max(3, int(dur / 0.1))
    pcmd = f"ip netns exec {PROBE[0]} ping -n -q -i 0.1 -W 1 -c {n} {ip_of(PROBE[1])}"
    pp = subprocess.Popen(shlex.split(pcmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    thr = {}
    for f, p in procs.items():
        out, err = p.communicate(timeout=dur + 15)
        try:
            j = json.loads(out)
            thr[f] = j["end"]["sum_received"]["bits_per_second"] / 1e6
        except Exception:
            thr[f] = 0.0
    out, _ = pp.communicate(timeout=dur + 15)
    for p in bgp:
        try: p.wait(timeout=dur + 15)
        except Exception: p.kill()
    rtt = RTT_MAX
    for line in out.splitlines():
        if "rtt min/avg/max" in line:
            rtt = float(line.split("=")[1].split("/")[1])
    loss = 100.0
    for line in out.splitlines():
        if "packet loss" in line:
            loss = float(line.split("%")[0].split()[-1])
    if loss >= 99.0:
        rtt = RTT_MAX
    return thr, rtt, loss


def utility(thr, rtt, asleep):
    T = statistics.mean(min(thr[f] / FLOWS[f][2], 1.0) for f in FLOWS)
    L = max(0.0, 1.0 - rtt / RTT_MAX)
    E = energy_score(asleep) / 0.5  # at most 4 of 8 links can sleep -> scale to [0,1]
    return WEIGHTS["T"] * T + WEIGHTS["L"] * L + WEIGHTS["E"] * E, T, L, E


def test(x, dur=2.0, settle=0.3, bg=None, rng=random):
    if bg is None:
        bg = {b: rng.choice(BG_RATES) for b in BG}
    paths, asleep = install(x)
    time.sleep(settle)
    thr, rtt, loss = measure(dur, bg)
    u, T, L, E = utility(thr, rtt, asleep)
    return {"x": "".join(map(str, x)), "u": u, "T": T, "L": L, "E": E, "rtt": rtt,
            "loss": loss, "thr": thr, "bg": bg, "dur": dur, "t": time.time()}


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(); print("built", len(PORTS), "ports")
    elif cmd == "teardown":
        teardown()
    elif cmd == "test":
        x = [int(c) for c in sys.argv[2]]
        dur = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
        print(json.dumps(test(x, dur)))
