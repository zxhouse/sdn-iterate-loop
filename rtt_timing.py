"""How much of the probe's mean RTT is measured after the TCP/UDP transfers have ended?
Repeats tests of random configurations on the live small testbed with the same timing as the
bank (transfers 1 s, probe 15 pings at 0.1 s), but records the send time and RTT of every
probe and the end of every transfer. Output: data/rtt_timing.jsonl."""
import json, random, shlex, subprocess, sys, threading, time
import numpy as np
import testbed as tb


def timed_measure(dur, bg):
    ends, procs, bgp = {}, {}, []
    def wait(name, p):
        out, _ = p.communicate(timeout=dur + 15); ends[name] = time.time(); procs[name] = out
    t_launch = time.time(); ths = []
    for b, r in bg.items():
        if r > 0:
            src, dst, port, _ = tb.BG[b]
            p = subprocess.Popen(shlex.split(f"ip netns exec {src} iperf3 -u -c {tb.ip_of(dst)} -p {port} -t {dur} -b {r}M -l 1200"),
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            ths.append(threading.Thread(target=wait, args=(f"bg{b}", p)))
    for f, (src, dst, d, port, P) in tb.FLOWS.items():
        p = subprocess.Popen(shlex.split(f"ip netns exec {src} iperf3 -c {tb.ip_of(dst)} -p {port} -t {dur} -l 16K -J -O 0 --connect-timeout 1000"),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ths.append(threading.Thread(target=wait, args=(f, p)))
    n = max(3, int(dur / 0.1))
    pp = subprocess.Popen(shlex.split(f"ip netns exec {tb.PROBE[0]} ping -D -n -i 0.1 -W 1 -c {n} {tb.ip_of(tb.PROBE[1])}"),
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for t in ths: t.start()
    pout, _ = pp.communicate(timeout=dur + 15)
    for t in ths: t.join()
    replies = []
    for line in pout.splitlines():
        if line.startswith("[") and "icmp_seq=" in line and "time=" in line:
            ts = float(line[1:line.index("]")]); seq = int(line.split("icmp_seq=")[1].split()[0])
            rtt = float(line.split("time=")[1].split()[0]); replies.append((seq, ts - rtt / 1000, rtt))
    thr = {}
    for f in tb.FLOWS:
        try: thr[f] = json.loads(procs[f])["end"]["sum_received"]["bits_per_second"] / 1e6
        except Exception: thr[f] = 0.0
    tcp_end = max(ends[f] for f in tb.FLOWS)
    return {"t_launch": t_launch, "tcp_end": tcp_end, "first_tcp_end": min(ends[f] for f in tb.FLOWS),
            "bg_end": max([ends[k] for k in ends if k.startswith("bg")], default=None), "n_sent": n,
            "replies": replies, "thr": thr}


if __name__ == "__main__":
    rng = random.Random(11); n_cfg = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    cfgs = [[rng.random() < 0.5 for _ in range(tb.NF)] for _ in range(n_cfg)]
    with open("data/rtt_timing.jsonl", "a") as log:
        for rep in range(2):
            for x in cfgs:
                x = [int(v) for v in x]; tb.install(x); time.sleep(0.3)
                bg = {b: rng.choice(tb.BG_RATES) for b in tb.BG}
                r = timed_measure(1.5, bg); r.update({"x": "".join(map(str, x)), "bg": bg, "rep": rep})
                log.write(json.dumps(r) + "\n"); log.flush()
    print("done")
