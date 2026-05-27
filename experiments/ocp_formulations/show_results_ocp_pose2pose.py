import numpy as np

save_path = "/home/sonia/Projects/arena-framework/experiments/ocp_formulations/unicycle_sweep_results.npy"

results = np.load(save_path, allow_pickle=True).item()

print("\n" + "=" * 80)
print("SAVED MOTION PRIMITIVE SEQUENCES")
print("=" * 80)

for key, data in results.items():

    theta0_deg = data["theta0_deg"]
    thetaf_deg = data["thetaf_deg"]

    sequence_string = " - ".join(data["sequence"])

    print(
        f"\n{theta0_deg:6.1f}° -> {thetaf_deg:6.1f}°"
    )

    print(f"Sequence: {sequence_string}")

    for p in data["primitives_with_info"]:

        if p["label"] in ["ArcL", "ArcR"]:

            delta_theta_deg = np.rad2deg(p["delta_theta"])

            print(
                f"  {p['label']}: "
                f"duration = {p['duration']:.4f} s, "
                f"delta_theta = {p['delta_theta']:.4f} rad "
                f"({delta_theta_deg:.2f} deg)"
            )