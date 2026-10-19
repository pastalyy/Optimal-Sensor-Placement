"""
units.py - real-world units for the three objectives (prices + display helpers).

Cost         -> Australian dollars per INSTALLED sensor (hardware + install + comms)
Coverage     -> % of the network (manholes) upstream of at least one sensor
Search cost  -> bits (log2 of the area to search);  2**bits = typical manholes to search

The prices below are ESTIMATES, not quotes. They are placeholders built from public
catalogue ranges and one utility rollout figure; treat them as stated assumptions in the
report and change them in one place (here) or from the app sidebar.

  Hardware (catalogue ranges found in the literature/price search):
      float / pulse switch   about $19 - $300          -> central estimate  $150
      continuous level sensor about $365 - $1,519+     -> central estimate  $900
  Install + communications (same for both types: site visit, mounting, telemetry unit,
  data plan). Sydney Water's rollout works out at roughly $1,000 per device ALL-IN
  ($15M / 15,000 devices), so $1,000 is used as the shared installed overhead.

  => installed price per sensor:  pulse ~ $1,150,   level ~ $1,900   (ratio about 1 : 1.65)
"""

HARDWARE_AUD = {1: 150.0, 2: 900.0}      # 1 = float/pulse, 2 = level
INSTALL_AND_COMMS_AUD = 1000.0           # shared per sensor, either type


def sensor_price(sensor_type, hardware=None, install=None):
    """Installed price (AUD) of one sensor of the given type."""
    hw = HARDWARE_AUD if hardware is None else hardware
    inst = INSTALL_AND_COMMS_AUD if install is None else install
    return float(hw[sensor_type] + inst)


def default_prices():
    return {0: 0.0, 1: sensor_price(1), 2: sensor_price(2)}


def apply_prices(objective_module, prices):
    """Set the cost table used by objective.type_weighted_cost (in place)."""
    for t, p in prices.items():
        objective_module.SENSOR_COST[t] = float(p)


def fmt_money(x):
    return f"A${x:,.0f}"


def describe_solution(chromosome, objectives, upstream_set, node_num):
    """Turn one solution's raw objective numbers into readable quantities."""
    import numpy as np
    sensor_nodes = np.where(np.asarray(chromosome) > 0)[0]
    covered = set()
    for n in sensor_nodes:
        covered |= upstream_set[int(n)]
    bits = float(objectives[2])
    return dict(
        cost_aud=float(objectives[0]),
        n_pulse=int((np.asarray(chromosome) == 1).sum()),
        n_level=int((np.asarray(chromosome) == 2).sum()),
        coverage_pct=100.0 * len(covered) / node_num,                  # % of network monitored
        effective_coverage_pct=100.0 * (-float(objectives[1])) / node_num,  # information-weighted
        search_bits=bits,
        search_manholes=2.0 ** bits,                                    # typical area to search
    )


def summary_line(d):
    return (f"cost={fmt_money(d['cost_aud'])}, coverage={d['coverage_pct']:.1f}% of network "
            f"(information-weighted {d['effective_coverage_pct']:.1f}%), "
            f"search effort ~{d['search_manholes']:.0f} manholes ({d['search_bits']:.2f} bits)")


EXPLANATION = """
**Total cost (A$)** - installed price of every sensor in the solution (hardware + installation +
communications). Estimates, not quotes.

**Coverage (%)** - share of the network (manholes) that sits upstream of at least one sensor, so a
detection there can be seen. Higher is better.

**Information-weighted coverage (%)** - the same, but each covered manhole counts by how informative its
best sensor is (level = 1.0, float/pulse = 0.5 by default). This is what the optimiser maximises.

**Search effort (manholes)** - after an alert, roughly how many manholes you would need to search to find
the source (2 to the power of the search cost in bits). Lower is better.
"""