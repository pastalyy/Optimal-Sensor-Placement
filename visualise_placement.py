"""
Run the NSGA-II sensor placement algorithm and save all outputs into a
single, organised folder for that run.

Usage examples:
    # Mixed (both types available) - just give both budgets:
    python3 run_and_visualise.py --pulsemax 20 --levelmax 5

    # Pulse-only (only give --pulsemax):
    python3 run_and_visualise.py --pulsemax 20

    # Level-only (only give --levelmax):
    python3 run_and_visualise.py --levelmax 10

Sensor types are inferred automatically from which budgets you provide -
no need to separately specify --sensortypes.

Output structure created for each run:
    OUTPUT/<run_name>/
        results/            <- NSGA-II convergence / Pareto plots
        network_diagram/    <- geographic map of the chosen sensor placement
"""
import argparse
import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt

from problem import Problem
from evolution import Evolution
import objective

# ---------------------------------------------------------------------------
# 1. Parse arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Run sensor placement optimisation and visualise the results")
parser.add_argument("--pulsemax", type=int, default=None, help="Max number of float/pulse sensors available")
parser.add_argument("--levelmax", type=int, default=None, help="Max number of level sensors available")
parser.add_argument("--iter", type=int, default=20, help="Number of NSGA-II generations")
parser.add_argument("--netdir", type=str, default="DATA/real_life_case_network/prepared/", help="Path to prepared network data")
parser.add_argument("--pick", type=str, default="median", choices=["median", "cheapest", "most_coverage"],
                     help="Which front-0 solution to visualise on the network map")
parser.add_argument("--outdir", type=str, default="OUTPUT", help="Parent folder for run outputs")
parser.add_argument("--name", type=str, default=None, help="Custom name for this run's output folder (auto-generated if omitted)")
args = parser.parse_args()

if args.pulsemax is None and args.levelmax is None:
    parser.error("You must provide at least one of --pulsemax or --levelmax")

# Infer allowed sensor types from which budgets were given
allowed_types = []
if args.pulsemax is not None:
    allowed_types.append(1)
if args.levelmax is not None:
    allowed_types.append(2)
allowed_types = tuple(allowed_types)

type_upbounds = {}
if 1 in allowed_types:
    type_upbounds[1] = args.pulsemax
if 2 in allowed_types:
    type_upbounds[2] = args.levelmax

# ---------------------------------------------------------------------------
# 2. Set up this run's output folder structure
# ---------------------------------------------------------------------------
if args.name:
    run_name = args.name
else:
    parts = []
    if args.pulsemax is not None:
        parts.append(f"pulse{args.pulsemax}")
    if args.levelmax is not None:
        parts.append(f"level{args.levelmax}")
    parts.append(f"iter{args.iter}")
    run_name = "_".join(parts)

run_dir = os.path.abspath(args.outdir)
results_dir = os.path.join(run_dir, "results", run_name)
network_dir_out = os.path.join(run_dir, "network_diagram", run_name)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(network_dir_out, exist_ok=True)
print(f"Working directory: {os.getcwd()}")
print(f"Results will be saved to:         {results_dir}")
print(f"Network diagram will be saved to: {network_dir_out}")

# ---------------------------------------------------------------------------
# 3. Load network + real node coordinates
# ---------------------------------------------------------------------------
print("Loading network data...")
G = nx.read_gpickle(args.netdir + "pipe_TM_clean_relabel.pkl")
with open(args.netdir + "upstream_set.pkl", "rb") as f:
    upstream_set = pickle.load(f)
with open(args.netdir + "upstream_arr.pkl", "rb") as f:
    upstream_arr = pickle.load(f)
with open(args.netdir + "conn_dict.pkl", "rb") as f:
    conn_dict = pickle.load(f)
with open(args.netdir + "label_mapping.pkl", "rb") as f:
    label_mapping = pickle.load(f)  # (x, y) coordinate -> node id

coord_of_node = {node_id: coord for coord, node_id in label_mapping.items()}
node_num = G.number_of_nodes()

# ---------------------------------------------------------------------------
# 4. Run NSGA-II (its own convergence plots get saved into results_dir)
# ---------------------------------------------------------------------------
print(f"Running NSGA-II ({args.iter} generations, sensor types={allowed_types}, budgets={type_upbounds})...")

problem = Problem(
    objectives=[objective.type_weighted_cost, objective.type_weighted_coverage, objective.final_edition_search_cost],
    node_num=node_num, upstream_arr=upstream_arr, upstream_set=upstream_set,
    sensor_upbound=max(type_upbounds.values()), coverage_constrain=0, graph=G, priority=None,
    conn_dict=conn_dict, allowed_types=allowed_types, type_upbounds=type_upbounds,
)

evo = Evolution(problem, node_num=node_num, num_of_generations=args.iter,
                 num_of_individuals=200, num_of_tour_particips=2,
                 tournament_prob=0.8, crossover_param=0.8, mutation_param=0.2,
                 input_fn='', fig_path=results_dir + os.sep)

final_population = evo.evolve()

# ---------------------------------------------------------------------------
# 5. Pick a solution and plot the network diagram
# ---------------------------------------------------------------------------
front0 = [ind for ind in final_population if ind in final_population.fronts[0]] \
    if hasattr(final_population, "fronts") else list(final_population)

front0_sorted = sorted(front0, key=lambda ind: ind.objectives[0])
if args.pick == "cheapest":
    chosen = front0_sorted[0]
elif args.pick == "most_coverage":
    chosen = min(front0, key=lambda ind: ind.objectives[1])
else:
    chosen = front0_sorted[len(front0_sorted) // 2]

cost, neg_cov, search_cost = chosen.objectives
n_pulse = int((chosen.chromosome == 1).sum())
n_level = int((chosen.chromosome == 2).sum())
print(f"\nChosen solution ({args.pick}): cost={cost:.1f}, coverage={-neg_cov:.1f}, "
      f"search_cost={search_cost:.3f}, {n_pulse} pulse sensors, {n_level} level sensors")

print("Plotting network diagram...")
fig, ax = plt.subplots(figsize=(12, 10))

for u, v in G.edges():
    if u in coord_of_node and v in coord_of_node:
        x1, y1 = coord_of_node[u]
        x2, y2 = coord_of_node[v]
        ax.plot([x1, x2], [y1, y2], color="lightgray", linewidth=0.4, zorder=1)

xs = [coord_of_node[n][0] for n in G.nodes() if n in coord_of_node]
ys = [coord_of_node[n][1] for n in G.nodes() if n in coord_of_node]
ax.scatter(xs, ys, s=2, color="gray", alpha=0.3, zorder=2, label="Network nodes")

pulse_nodes = [i for i in range(node_num) if chosen.chromosome[i] == 1]
level_nodes = [i for i in range(node_num) if chosen.chromosome[i] == 2]
pulse_xy = [coord_of_node[n] for n in pulse_nodes if n in coord_of_node]
level_xy = [coord_of_node[n] for n in level_nodes if n in coord_of_node]

if pulse_xy:
    px, py = zip(*pulse_xy)
    ax.scatter(px, py, s=60, color="orange", edgecolor="black", zorder=3,
               label=f"Float/pulse sensor (n={len(pulse_xy)})")
if level_xy:
    lx, ly = zip(*level_xy)
    ax.scatter(lx, ly, s=90, color="green", marker="^", edgecolor="black", zorder=4,
               label=f"Level sensor (n={len(level_xy)})")

ax.set_title(f"Sensor Placement on Real HK Sewage Network\n"
             f"cost={cost:.1f}, coverage={-neg_cov:.1f}, search_cost={search_cost:.3f}")
ax.legend(loc="upper right")
ax.set_xlabel("X coordinate")
ax.set_ylabel("Y coordinate")
plt.tight_layout()

network_map_path = os.path.join(network_dir_out, "network_map.png")
plt.savefig(network_map_path, dpi=150)

print(f"\nAll outputs for this run:")
print(f"  Results:         {results_dir}")
print(f"  Network diagram: {network_dir_out}")