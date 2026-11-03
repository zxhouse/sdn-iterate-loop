"""Iterate-test loop and baseline policies for configuration search on an SDN
data plane. Semantics of `loop` follow the reference engine of the base paper
(dt-formalisation/code/engine.py, non-legacy mode); what changes is the world:
tests are drawn from measured (testbed) or modelled (topology) outcome banks and
ideas are pre-screened by a network model instead of an oracle-plus-noise."""
import numpy as np
from collections import deque

PRIOR_MU, PRIOR_VAR = 0.5, 0.08


class Belief:
    __slots__ = ("mu", "var", "s2", "n")

    def __init__(self, sigma):
        self.mu, self.var, self.s2, self.n = PRIOR_MU, PRIOR_VAR, sigma ** 2, 0

    def update(self, y):
        nv = 1.0 / (1.0 / self.var + 1.0 / self.s2)
        self.mu = min(1.0, max(0.0, nv * (self.mu / self.var + y / self.s2)))
        self.var = nv; self.n += 1


class World:
    """U: true utility per code; draw(c, rng): one test; screen: twin estimate per code
    (None -> oracle-plus-noise screening as in the base paper); sigma: noise the team assumes."""

    def __init__(self, U, draw, sigma, screen=None, name=""):
        self.U = np.asarray(U, float); self.n = int(np.log2(self.U.size))
        self.ustar = float(self.U.max()); self.draw = draw; self.sigma = sigma
        self.screen = screen; self.name = name


def bank_world(samples, screen=None, name="testbed", extra_sd=0.0):
    """samples: list (per code) of arrays of measured utilities; truth = mean."""
    U = np.array([np.mean(s) for s in samples])
    sd = float(np.sqrt(np.mean([np.var(s, ddof=1) for s in samples])))
    def draw(c, rng):
        s = samples[c]
        return float(s[rng.integers(len(s))]) + (rng.normal(0, extra_sd) if extra_sd else 0.0)
    return World(U, draw, float(np.hypot(sd, extra_sd)), screen, name)


def matrix_world(M, screen=None, sigma_m=0.0, name=""):
    """M: (C, K) utilities under K traffic draws; test = random draw + N(0, sigma_m)."""
    U = M.mean(1); K = M.shape[1]
    sd_tm = float(np.sqrt(np.mean(M.var(1, ddof=1))))
    def draw(c, rng):
        return float(M[c, rng.integers(K)] + (rng.normal(0, sigma_m) if sigma_m else 0.0))
    return World(U, draw, float(np.hypot(sd_tm, sigma_m)), screen, name)


class Run:
    def __init__(self, w, seed, bias=0.0):
        self.w, self.rng = w, np.random.default_rng(seed)
        self.b, self.tests, self.bias = {}, 0, bias
        self.best_found, self.best_code = -1.0, -1
        self.n = w.n

    def rand(self):
        bits = self.rng.random(self.n) < 0.3
        return int(sum(1 << i for i in np.flatnonzero(bits)))

    def mutate(self, c, r_max=2):
        for i in self.rng.choice(self.n, size=self.rng.integers(1, r_max + 1), replace=False):
            c ^= (1 << int(i))
        return c

    def est(self, c, proxy_sd=0.10):
        if self.w.screen is None:
            return self.w.U[c] + self.rng.normal(0, proxy_sd)
        return self.w.screen[c]

    def test(self, c, bias=0.0):
        if c not in self.b:
            self.b[c] = Belief(self.w.sigma)
            if self.w.U[c] > self.best_found:
                self.best_found, self.best_code = self.w.U[c], c
        self.b[c].update(self.w.draw(c, self.rng) + bias); self.tests += 1

    def leader(self):
        return max(self.b, key=lambda c: self.b[c].mu)

    def out(self):
        L = self.leader(); us = self.w.ustar
        cau = max(self.b, key=lambda c: self.b[c].mu - np.sqrt(self.b[c].var))
        return {"shipped": self.w.U[L] / us, "found": self.best_found / us,
                "belief": self.b[L].mu / us, "cautious": self.w.U[cau] / us,
                "n_proto": len(self.b), "tests": self.tests, "inc": int(L), "found_code": int(self.best_code)}


def loop(w, budget=300, seed=0, n_new=1.0, n_top=3, n_rr=1, ideas=5, r_max=2, screen=True,
         bias=0.0, memory=None, alloc=None, start=None, trace=False):
    """One round = build `n_new` new prototypes (fractional allowed), re-test the
    `n_top` leaders, re-test one prototype in rotation. With `alloc` (tests per
    round, e.g. 5) the round spends exactly `alloc` tests: n_new new, the rest on
    the three leaders in turn, no rotation (as the allocation study of the base paper)."""
    R = Run(w, seed)
    cand, fresh, queue = {}, [], deque()

    def add(c):
        if c not in R.b and c not in cand:
            cand[c] = R.est(c); fresh.append(c)

    for c in (start if start is not None else [R.rand() for _ in range(5)]):
        add(c)
    k, tr = 0, []
    while R.tests < budget:
        fresh.clear()
        build = int(np.floor((k + 1) * n_new + 1e-9) - np.floor(k * n_new + 1e-9))
        if not R.b:
            build = max(build, 1)

        def ideate():
            if R.b:
                fr = sorted(R.b, key=lambda c: R.b[c].mu, reverse=True)[:3]
                base = fr[R.rng.integers(len(fr))]
            else:
                base = R.rand()
            for _ in range(ideas):
                add(R.mutate(base, r_max))
        ideate()
        tries = 0
        while len(cand) < build and tries < 50:
            ideate(); tries += 1
        built = 0
        for _ in range(build):
            if not cand or R.tests >= budget:
                break
            if screen:
                ch = max(cand, key=cand.get)
            else:
                pool = [c for c in fresh if c in cand] or list(cand)
                ch = pool[R.rng.integers(len(pool))]
            del cand[ch]
            R.test(ch); queue.append(ch); built += 1
        if alloc is not None:
            lead = sorted(R.b, key=lambda c: R.b[c].mu, reverse=True)[:3]
            for j in range(alloc - built):
                if R.tests >= budget: break
                rank = j % len(lead)
                R.test(lead[rank], bias if rank == 0 else 0.0)
        else:
            top = sorted(R.b, key=lambda c: R.b[c].mu, reverse=True)[:n_top]
            for rank, c in enumerate(top):
                if R.tests >= budget: break
                R.test(c, bias if rank == 0 else 0.0)
            for _ in range(n_rr):
                if R.tests >= budget or not queue: break
                c = queue.popleft()
                while c not in R.b and queue:
                    c = queue.popleft()
                if c in R.b:
                    queue.append(c); R.test(c)
        if memory is not None and len(R.b) > memory:
            keep = sorted(R.b, key=lambda c: R.b[c].mu, reverse=True)[:memory]
            R.b = {c: R.b[c] for c in keep}
            queue = deque(c for c in queue if c in R.b)
        if trace:
            o = R.out(); o["round"] = k; tr.append(o)
        k += 1
    o = R.out()
    if trace:
        o["trace"] = tr
    return o


# ------------------------------------------------------------------ baselines
def p_random(w, budget, seed, _=None):
    R = Run(w, seed); pool = [R.rand() for _ in range(5)]
    for c in pool: R.test(c)
    while R.tests < budget:
        c = R.mutate(pool[R.rng.integers(len(pool))]); R.test(c); pool.append(c)
    return R.out()


def p_greedy(w, budget, seed, reps=2):
    R = Run(w, seed); cur = R.rand()
    for _ in range(reps): R.test(cur)
    while R.tests < budget:
        best, bmu = None, R.b[cur].mu
        for i in R.rng.permutation(R.n):
            q = cur ^ (1 << int(i))
            for _ in range(reps):
                if R.tests < budget: R.test(q)
            if q in R.b and R.b[q].mu > bmu: best, bmu = q, R.b[q].mu
            if R.tests >= budget: break
        if best is None:
            cur = R.rand()
            if R.tests < budget: R.test(cur)
        else:
            cur = best
    return R.out()


def p_eps(w, budget, seed, eps=0.3):
    R = Run(w, seed)
    for _ in range(5): R.test(R.rand())
    while R.tests < budget:
        L = R.leader(); R.test(R.mutate(L) if R.rng.random() < eps else L)
    return R.out()


def _screened(R, c, ideas=5):
    cs = [R.mutate(c) for _ in range(ideas)]
    return max(cs, key=lambda q: R.est(q))


def p_ucb(w, budget, seed, par=(0.1, 3), screen=False):
    cx, every = par
    R = Run(w, seed)
    for _ in range(5): R.test(R.rand())
    step = 0
    while R.tests < budget:
        step += 1
        if step % every == 0:
            L = R.leader(); R.test(_screened(R, L) if screen else R.mutate(L)); continue
        lt = np.log(max(2, R.tests))
        R.test(max(R.b, key=lambda c: R.b[c].mu + cx * np.sqrt(lt / R.b[c].n)))
    return R.out()


def p_thompson(w, budget, seed, every=3):
    R = Run(w, seed)
    for _ in range(5): R.test(R.rand())
    step = 0
    while R.tests < budget:
        step += 1
        if step % every == 0:
            R.test(R.mutate(R.leader())); continue
        ks = list(R.b); mu = np.array([R.b[c].mu for c in ks]); sd = np.sqrt([R.b[c].var for c in ks])
        R.test(ks[int(np.argmax(mu + sd * R.rng.standard_normal(len(ks))))])
    return R.out()


def _twin_order(w, rng):
    """Configurations from the twin's best down; ties of the twin are broken at random."""
    return np.lexsort((rng.random(w.screen.size), -np.asarray(w.screen)))


def p_twin_only(w, budget, seed, _=None):
    """Deploys the twin's best configuration without testing. When several configurations
    share the twin's maximum, the value reported is the expectation over a random choice
    among them (and their minimum and maximum are returned as well)."""
    top = np.flatnonzero(np.isclose(w.screen, np.max(w.screen), rtol=0, atol=1e-12)); us = w.ustar
    v = float(np.mean(w.U[top])) / us
    return {"shipped": v, "found": v, "belief": np.nan, "cautious": v, "n_proto": 1, "tests": 0,
            "n_tied": int(top.size), "tied_min": float(w.U[top].min()) / us, "tied_max": float(w.U[top].max()) / us}


def p_twin_validate(w, budget, seed, k=10):
    """Simulate-then-validate: the twin's top-k configurations, tests spread evenly, ship best mean."""
    R = Run(w, seed); top = list(_twin_order(w, R.rng)[:k])
    i = 0
    while R.tests < budget:
        R.test(int(top[i % k])); i += 1
    return R.out()


def p_measure_once(w, budget, seed, screen=True):
    """Online local search that never re-tests: every test is a new variant of the
    best-measured configuration; ships the best single measurement."""
    return loop(w, budget, seed, n_new=5, alloc=5, screen=screen)


def p_loop(w, budget, seed, screen=True):
    return loop(w, budget, seed, screen=screen)


def p_loop_twinstart(w, budget, seed, _=None):
    start = list(_twin_order(w, np.random.default_rng(seed + 7))[:5])
    return loop(w, budget, seed, start=[int(c) for c in start])
