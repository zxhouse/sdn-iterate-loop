import subprocess, os, signal, time
import testbed as tb
out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
for line in out.splitlines()[1:]:
    pid, args = line.strip().split(None, 1)
    if args.startswith("iperf3 "):
        os.kill(int(pid), signal.SIGKILL)
time.sleep(1)
for f, (s, d, dm, p, P) in tb.FLOWS.items():
    tb.sh(f"ip netns exec {d} iperf3 -s -D -p {p}")
for b, (s, d, p, pp) in tb.BG.items():
    tb.sh(f"ip netns exec {d} iperf3 -s -D -p {p}")
print("servers restarted")
