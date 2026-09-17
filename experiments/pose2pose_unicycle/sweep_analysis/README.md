# Parameter-sweep analysis

Run these scripts with `arena-env`. From this folder:

```bash
python histogram_plot.py
python ecdf_plot.py
python sweep_statistics.py
python worst_cases_plot.py
```

- `histogram_plot.py`: full-range signed traversal-time difference histogram with a logarithmic count axis, mean and median markers.
- `ecdf_plot.py`: relative time-discrepancy and Hausdorff ECDFs for one to three sweeps, in orange, blue and green.
- `sweep_statistics.py`: mean, median, 95th percentile and maximum for absolute time error, relative time error, signed time difference and saved Hausdorff distance, using cases with all measurements. It also counts cases where the OCP is faster and, with multiple inputs, case IDs shared by all inputs. Use `--top-cases 3` to compare the largest time errors and Hausdorff distances by case ID.
- `short_distance_ocp_plot.py`: explores `D/R = 1, 2, 3` with start pose `(0, 0, 0)` and end pose `(0, D, pi)`, using `R=1`. Uses TST initialization for every case and shows the OCP result and its seed, with paths and controls. No analytical optimum is generated. Saves PDF, PNG and a JSON record of the solve for each case. Options: `--N 100 --M 4 --linear-solver mumps --no-show`. Add `--theta-f 0` for equal headings (start and end both zero); these outputs have a separate `_thetaf_0` suffix. The default `--theta-f pi` keeps opposing headings.
- `worst_cases_plot.py`: selects the three largest absolute time discrepancies and recomputes analytical and OCP solutions using saved case parameters and TST initialization with MUMPS. The combined figure has three case columns, with paths above and stacked velocity/angular-velocity comparisons below. Also saves a separate `_controls` figure. Controls use physical time and show the bounds. Defaults to the Sobol TST N100 sweep. Saves PDF, PNG, and a JSON summary in `../figures/`; accepts an alternative input path and `--no-show`.
- `selected_worst_cases_plot.py`: recomputes N300 cases 3395, 7337 and 4092, selected for the largest saved time difference, largest saved Hausdorff distance and membership in both top-three lists. Uses MUMPS by default and accepts `--linear-solver ma27` on a machine with MA27. Saves the same path/control layouts and a JSON summary with saved metrics and recomputed traversal times.

Histogram defaults to the Sobol TST N300 sweep, statistics to N100, and ECDF to N50, N100 and N300. Edit `RESULTS_FILENAME` / `RESULTS_FILES` or pass JSON paths:

```bash
python histogram_plot.py /path/to/sweep.json --no-show
python ecdf_plot.py /path/to/sweep1.json /path/to/sweep2.json --no-show
python sweep_statistics.py /path/to/sweep.json
python sweep_statistics.py sweep4/sobol_sweep_OCP_TST_initial_guess_N50_M4.json sweep4/sobol_sweep_OCP_TST_initial_guess_N100_M4.json sweep4/sobol_sweep_OCP_TST_initial_guess_N300_M4.json
```

Inputs can also be specified relative to `../results/`, such as `sweep4/sobol_sweep_OCP_TST_initial_guess_N100_M4.json`. Default paths work regardless of the working directory.

Plots are saved as PDF and PNG in `../figures/`. Histogram filenames derive from the input filename; ECDF defaults to `sweep_ecdf_comparison`, configurable with `--output-name`. Statistics print to the terminal. Use `--help` for all command-line options.

The ECDF plot uses relative time discrepancy `100 * abs(T_OCP - T_analytical) / T_analytical`, in percent.
The statistics script uses `abs(T_OCP - T_analytical)` for absolute time error and `100 * abs(T_OCP - T_analytical) / T_analytical` for relative time error. It reports `T_OCP - T_analytical` separately as the signed time difference. Its Hausdorff statistics use the values saved in the sweep JSON files.

For the worst-case comparison on a computer with MA27 installed, run
`python worst_cases_plot.py --linear-solver ma27`. MUMPS remains the default.
For the selected N300 cases, run
`python selected_worst_cases_plot.py --linear-solver ma27 --no-show`.
The N300 JSON must be available at
`../results/sweep4/sobol_sweep_OCP_TST_initial_guess_N300_M4.json`;
results files are ignored by Git and need to be transferred separately.
The solver is recorded in output filenames and the JSON summary; figure legends
simply say “OCP solution.” Analytical paths include initial, final, and intermediate
pose markers and heading arrows.

- `short_distance_heading_paths.py`: path-only OCP comparison for `D/R = 1, 2, 3` and every pair of headings in `{0, pi/2, pi, 3pi/2}` (48 cases). All cases use TST initialization, `R=1`, and positions `(0,0)` to `(0,D)`. The default output is one 4-by-4 grid with three colored paths per panel; `--layout separate` generates one grid per distance. PDF/PNG figures and an incremental JSON cache are saved in `figures/`. Use `--recompute` to replace cached solves, or change `--N`, `--M`, and `--linear-solver`. Failed solves are labeled and their paths omitted. Example: `python experiments/pose2pose_unicycle/sweep_analysis/short_distance_heading_paths.py --no-show`.
