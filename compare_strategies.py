"""Compare mixed vs single-sensor-type Pareto fronts from the three runs."""
import pickle
import matplotlib.pyplot as plt

RUNS = {
    "Mixed (float/pulse + level)": "TESTOUTPUT/compare_mixed/gen_20_mp0.20_cp0.80_up100_cc0/0/solution.pkl",
    "Float/pulse only": "TESTOUTPUT/compare_pulse_only/gen_20_mp0.20_cp0.80_up100_cc0/0/solution.pkl",
    "Level sensors only": "TESTOUTPUT/compare_level_only/gen_20_mp0.20_cp0.80_up100_cc0/0/solution.pkl",
}

plt.figure(figsize=(8, 6))
colors = {"Mixed (float/pulse + level)": "tab:blue",
          "Float/pulse only": "tab:orange",
          "Level sensors only": "tab:green"}

results = {}
for label, path in RUNS.items():
    with open(path, "rb") as f:
        sol = pickle.load(f)
    last_gen = max(sol.keys())
    front0 = sol[last_gen]
    costs = [s[0] for s in front0]
    coverage = [-s[1] for s in front0]
    results[label] = front0
    plt.scatter(costs, coverage, alpha=0.6, label=label, color=colors[label])

plt.xlabel("Total Cost")
plt.ylabel("Weighted Coverage (information value)")
plt.title("Final Pareto Fronts (gen 20): Mixed vs Single Sensor-Type Strategies\nReal Hong Kong Sewage Network (4,394 nodes)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("comparison_mixed_vs_single.png", dpi=150)
print("Saved: comparison_mixed_vs_single.png")

# Print a quick summary: for a given cost budget, what's the best coverage each strategy achieves?
print("\nBest coverage achieved within cost budgets:")
for budget in [10, 20, 30, 50, 80]:
    print(f"\n  Cost budget <= {budget}:")
    for label, front0 in results.items():
        eligible = [(-s[1]) for s in front0 if s[0] <= budget]
        best = max(eligible) if eligible else 0
        print(f"    {label:35s}: best coverage = {best:.1f}")