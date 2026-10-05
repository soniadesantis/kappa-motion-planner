


def compute_trajectory_bicycle(
    corridor_list,
    initial_pose,
    final_pose,
    bicycle,
):
    # ---------------------------------------------------------------
    # 1. Construct validated baseline
    # ---------------------------------------------------------------
    baseline = compute_bicycle_baseline(
        corridor_list=corridor_list,
        bicycle=bicycle,
    )

    if baseline is None:
        return None

    # ---------------------------------------------------------------
    # 2. Refine internal path
    # ---------------------------------------------------------------
    refinement = refine_bicycle_baseline(
        corridor_list=corridor_list,
        bicycle=bicycle,
        baseline=baseline,
    )

    # If refinement fails, retain the validated baseline
    if refinement is None:
        internal_path = baseline
    else:
        internal_path = refinement

    # ---------------------------------------------------------------
    # 3. Construct boundary-pose connections
    # ---------------------------------------------------------------
    boundary_connections = compute_bicycle_boundary_connections(
        corridor_list=corridor_list,
        bicycle=bicycle,
        initial_pose=initial_pose,
        final_pose=final_pose,
        internal_path=internal_path,
    )

    if boundary_connections is None:
        return None

    # ---------------------------------------------------------------
    # 4. Assemble complete trajectory
    # ---------------------------------------------------------------
    trajectory = assemble_bicycle_trajectory(
        internal_path=internal_path,
        boundary_connections=boundary_connections,
    )

    return trajectory