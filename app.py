"""
Interactive dashboard for SA Water sensor placement optimisation.

Run with:
    streamlit run app.py

Lets you set independent budgets for float/pulse and level sensors,
run the NSGA-II optimisation live, and see the results as an
interactive network map plus a Pareto front plot - all without
touching the command line for each new scenario.
"""
import random as pyrandom

import numpy as np
import pickle
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from problem import Problem
from evolution import Evolution
import objective
import units

st.set_page_config(page_title="Sensor Placement Optimisation", layout="wide")

NETDIR = "DATA/real_life_case_network/prepared/"


# ---------------------------------------------------------------------------
# Cached network loading - only happens once, not on every button click
# ---------------------------------------------------------------------------
@st.cache_resource
def load_network():
    G = nx.read_gpickle(NETDIR + "pipe_TM_clean_relabel.pkl")
    with open(NETDIR + "upstream_set.pkl", "rb") as f:
        upstream_set = pickle.load(f)
    with open(NETDIR + "upstream_arr.pkl", "rb") as f:
        upstream_arr = pickle.load(f)
    with open(NETDIR + "conn_dict.pkl", "rb") as f:
        conn_dict = pickle.load(f)
    with open(NETDIR + "label_mapping.pkl", "rb") as f:
        label_mapping = pickle.load(f)
    coord_of_node = {node_id: coord for coord, node_id in label_mapping.items()}
    return G, upstream_set, upstream_arr, conn_dict, coord_of_node


G, upstream_set, upstream_arr, conn_dict, coord_of_node = load_network()
node_num = G.number_of_nodes()


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("Sensor Budget")
pulse_enabled = st.sidebar.checkbox("Use float/pulse sensors", value=True)
pulse_max = st.sidebar.number_input("Max float/pulse sensors", min_value=1, max_value=200, value=20,
                                     disabled=not pulse_enabled)

level_enabled = st.sidebar.checkbox("Use level sensors", value=True)
level_max = st.sidebar.number_input("Max level sensors", min_value=1, max_value=200, value=5,
                                     disabled=not level_enabled)

st.sidebar.title("Sensor Prices (AUD, installed)")
_defaults = units.default_prices()
pulse_price = st.sidebar.number_input("Float/pulse sensor", min_value=1, max_value=100000,
                                      value=int(_defaults[1]), step=50)
level_price = st.sidebar.number_input("Level sensor", min_value=1, max_value=100000,
                                      value=int(_defaults[2]), step=50)

st.sidebar.title("Algorithm Settings")
generations = st.sidebar.slider("Generations", min_value=5, max_value=100, value=20, step=5)
pick_mode = st.sidebar.selectbox("Solution to display", ["median", "cheapest", "most_coverage"])
seed = st.sidebar.number_input("Random seed (for reproducibility)", min_value=0, max_value=99999, value=42)

run_button = st.sidebar.button("Run optimisation", type="primary")

st.sidebar.markdown("---")
st.sidebar.caption(
    "Network: real Hong Kong sewage network (Tuen Mun District), "
    "4,394 nodes, 4,308 edges. Source: Wang, Xu & Zhou (2024), "
    "*Sustainable Cities and Society*."
)


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Mixed Sensor Type Placement Optimisation")
st.caption(
    "Optimises placement of float/pulse and level "
    " sensors jointly, using a modified version of "
    "Wang et al. (2024)'s NSGA-II sensor placement algorithm."
)

if not pulse_enabled and not level_enabled:
    st.warning("Enable at least one sensor type in the sidebar to run the optimisation.")
    st.stop()

if not run_button and "last_result" not in st.session_state:
    st.info("Set your sensor budgets in the sidebar, then click **Run optimisation** to begin.")
    st.stop()


def run_optimisation():
    units.apply_prices(objective, {1: pulse_price, 2: level_price})
    pyrandom.seed(seed)
    np.random.seed(seed)

    allowed_types = []
    type_upbounds = {}
    if pulse_enabled:
        allowed_types.append(1)
        type_upbounds[1] = pulse_max
    if level_enabled:
        allowed_types.append(2)
        type_upbounds[2] = level_max
    allowed_types = tuple(allowed_types)

    problem = Problem(
        objectives=[objective.type_weighted_cost, objective.type_weighted_coverage, objective.final_edition_search_cost],
        node_num=node_num, upstream_arr=upstream_arr, upstream_set=upstream_set,
        sensor_upbound=max(type_upbounds.values()), coverage_constrain=0, graph=G, priority=None,
        conn_dict=conn_dict, allowed_types=allowed_types, type_upbounds=type_upbounds,
    )

    evo = Evolution(problem, node_num=node_num, num_of_generations=generations,
                     num_of_individuals=200, num_of_tour_particips=2,
                     tournament_prob=0.8, crossover_param=0.8, mutation_param=0.2,
                     input_fn='', fig_path='./streamlit_tmp/')

    final_population = evo.evolve()

    front0 = [ind for ind in final_population if ind in final_population.fronts[0]] \
        if hasattr(final_population, "fronts") else list(final_population)

    return front0


if run_button:
    with st.spinner(f"Running NSGA-II for {generations} generations ..."):
        import os
        os.makedirs("./streamlit_tmp/", exist_ok=True)
        front0 = run_optimisation()
        st.session_state["last_result"] = front0
        st.session_state["prices_used"] = (pulse_price, level_price)
        st.session_state["run_id"] = st.session_state.get("run_id", 0) + 1

front0 = st.session_state["last_result"]

front0_sorted = sorted(front0, key=lambda ind: ind.objectives[0])
if pick_mode == "cheapest":
    chosen = front0_sorted[0]
elif pick_mode == "most_coverage":
    chosen = min(front0, key=lambda ind: ind.objectives[1])
else:
    chosen = front0_sorted[len(front0_sorted) // 2]
shown_how = f"'{pick_mode}' pick"

# A point clicked on the Pareto plot overrides the pick mode. The chart's key includes the run number
# and pick mode, so starting a new run or changing the pick mode clears the click.
PARETO_KEY = f"pareto_{st.session_state.get('run_id', 0)}_{pick_mode}"


def clicked_solution_index(event, n_solutions):
    """Index into front0 of the clicked Pareto point, or None."""
    try:
        points = event["selection"]["points"]
    except (TypeError, KeyError):
        return None
    for p in points:
        if p.get("curve_number", 0) != 0:
            continue
        idx = p.get("point_index", p.get("point_number"))
        if idx is not None and 0 <= int(idx) < n_solutions:
            return int(idx)
    return None


_clicked = clicked_solution_index(st.session_state.get(PARETO_KEY), len(front0))
if _clicked is not None:
    chosen = front0[_clicked]
    shown_how = "solution you clicked on the Pareto plot"

d = units.describe_solution(chosen.chromosome, chosen.objectives, upstream_set, node_num)
n_pulse, n_level = d["n_pulse"], d["n_level"]
pp, lp = st.session_state.get("prices_used", (pulse_price, level_price))

# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total cost", units.fmt_money(d["cost_aud"]),
            help=f"{n_pulse} x A${pp:,.0f} + {n_level} x A${lp:,.0f} (installed prices used for this run)")
col2.metric("Network covered", f"{d['coverage_pct']:.1f}%",
            help="Share of manholes upstream of at least one sensor")
_covered_nodes = d["coverage_pct"] / 100.0 * node_num
_share = (100.0 * d["search_manholes"] / _covered_nodes) if _covered_nodes else 0.0
col3.metric("Candidate locations per alert", f"~{d['search_manholes']:.0f}",
            help=f"After an alert, the source could be in roughly this many manholes = 2^{d['search_bits']:.2f} "
                 f"({d['search_bits']:.2f} bits), about {_share:.0f}% of the monitored network. Lower is better.")
col4.metric("Pulse sensors", n_pulse)
col5.metric("Level sensors", n_level)
st.caption(f"Showing: {shown_how}. "
           f"Information-weighted coverage: {d['effective_coverage_pct']:.1f}% of network.")


tab1, tab2 = st.tabs(["Network Map", "Pareto Front"])

# ---------------------------------------------------------------------------
# Tab 1: interactive network map
# ---------------------------------------------------------------------------
with tab1:
    edge_x, edge_y = [], []
    for u, v in G.edges():
        if u in coord_of_node and v in coord_of_node:
            x1, y1 = coord_of_node[u]
            x2, y2 = coord_of_node[v]
            edge_x += [x1, x2, None]
            edge_y += [y1, y2, None]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="lightgray", width=0.5),
        hoverinfo="skip", showlegend=False,
    ))

    all_xs = [coord_of_node[n][0] for n in G.nodes() if n in coord_of_node]
    all_ys = [coord_of_node[n][1] for n in G.nodes() if n in coord_of_node]
    fig.add_trace(go.Scattergl(
        x=all_xs, y=all_ys, mode="markers",
        marker=dict(size=3, color="lightgray", opacity=0.5),
        name="Network nodes", hoverinfo="skip",
    ))

    pulse_nodes = [i for i in range(node_num) if chosen.chromosome[i] == 1]
    level_nodes = [i for i in range(node_num) if chosen.chromosome[i] == 2]

    if pulse_nodes:
        px = [coord_of_node[n][0] for n in pulse_nodes if n in coord_of_node]
        py = [coord_of_node[n][1] for n in pulse_nodes if n in coord_of_node]
        fig.add_trace(go.Scattergl(
            x=px, y=py, mode="markers",
            marker=dict(size=11, color="orange", line=dict(width=1, color="black")),
            name=f"Float/pulse sensor (n={len(px)})",
            hovertemplate="Pulse sensor<br>x=%{x}<br>y=%{y}<extra></extra>",
        ))

    if level_nodes:
        lx = [coord_of_node[n][0] for n in level_nodes if n in coord_of_node]
        ly = [coord_of_node[n][1] for n in level_nodes if n in coord_of_node]
        fig.add_trace(go.Scattergl(
            x=lx, y=ly, mode="markers",
            marker=dict(size=13, color="green", symbol="triangle-up", line=dict(width=1, color="black")),
            name=f"Level sensor (n={len(lx)})",
            hovertemplate="Level sensor<br>x=%{x}<br>y=%{y}<extra></extra>",
        ))

    fig.update_layout(
        height=700,
        xaxis=dict(title="X coordinate", scaleanchor="y"),
        yaxis=dict(title="Y coordinate"),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Zoom, pan, and hover over sensors to inspect this solution's placement.")

# ---------------------------------------------------------------------------
# Tab 2: Pareto front
# ---------------------------------------------------------------------------
with tab2:
    costs = [ind.objectives[0] for ind in front0]
    covs = [100.0 * (-ind.objectives[1]) / node_num for ind in front0]
    search = [2.0 ** ind.objectives[2] for ind in front0]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=costs, y=covs, mode="markers",
        marker=dict(size=9, color="steelblue"),
        name="Pareto front solutions",
        customdata=search,
        hovertemplate="Cost=A$%{x:,.0f}<br>Info-weighted coverage=%{y:.1f}%<br>"
                      "Search effort=~%{customdata:.0f} manholes<extra></extra>",
    ))
    fig2.add_trace(go.Scatter(
        x=[d["cost_aud"]], y=[d["effective_coverage_pct"]], mode="markers",
        marker=dict(size=16, color="red", symbol="star"),
        name="Displayed solution",
    ))
    fig2.update_layout(
        height=600,
        xaxis_title="Total cost (A$, installed)",
        yaxis_title="Information-weighted coverage (% of network)",
        margin=dict(l=10, r=10, t=10, b=10),
    )
    try:
        st.plotly_chart(fig2, use_container_width=True, on_select="rerun",
                        selection_mode="points", key=PARETO_KEY)
        st.caption(f"{len(front0)} solutions on the final Pareto front. "
                   f"**Click a point to show that solution on the Network Map tab.** "
                   f"The red star is the solution currently shown. Hover a point to see its candidate locations.")
    except TypeError:       # older Streamlit without click selection (needs >= 1.35)
        st.plotly_chart(fig2, use_container_width=True)
        st.warning("Click-to-select needs Streamlit 1.35 or newer (`pip install -U streamlit`).")