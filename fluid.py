"""Flow-level (fluid) model of an SDN data plane: max-min fair sharing of link
capacity among elastic flows (TCP-like) with per-flow rate caps (access rate or
OpenFlow meter) and non-responsive background load. Batched over traffic draws."""
import numpy as np


def maxmin(A, cap, fcap):
    """A: (L, F) 0/1 incidence; cap: (B, L) residual capacities; fcap: (B, F) flow caps.
    Returns rates (B, F) by progressive filling. Flows with no links are capped only by fcap."""
    B, L = cap.shape
    F = A.shape[1]
    rate = np.zeros((B, F))
    active = np.ones((B, F), dtype=bool)
    rem = cap.astype(float).copy()
    Af = A.astype(float)
    for _ in range(F + 1):
        if not active.any():
            break
        nact = active.astype(float) @ Af.T                      # (B, L) active flows per link
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(nact > 0, rem / nact, np.inf)        # fair increment per link
        # per-flow bottleneck increment: min over its links, and its own cap headroom
        inc_link = np.where(A[None, :, :] > 0, share[:, :, None], np.inf).min(axis=1)  # (B, F)
        inc = np.minimum(inc_link, fcap - rate)
        inc = np.where(active, inc, np.inf)
        d = inc.min(axis=1)                                       # (B,)
        d = np.where(np.isfinite(d), d, 0.0)
        rate += np.where(active, d[:, None], 0.0)
        rem -= (np.where(active, d[:, None], 0.0)) @ Af.T
        # freeze flows at their cap or crossing a saturated link
        sat = rem <= 1e-9 * np.maximum(cap, 1)
        on_sat = (sat.astype(float) @ Af) > 0                     # (B, F)
        capped = rate >= fcap - 1e-9
        active &= ~(on_sat | capped)
    return rate
