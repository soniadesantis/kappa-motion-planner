# Short thesis chapter draft

- `chapter_kappa_planner.tex`: short chapter connecting the analytical methods,
  software components, and one example.
- `appendix_kappa_installation.tex`: installation instructions, separated from
  the research chapter.
- `kappa_example.py`: runnable example included directly in the chapter.
- `preview.tex`: standalone wrapper, numbering the chapter as Chapter 7.
- `preview.pdf`: compiled reading copy.

Compile from this directory with `pdflatex preview.tex` (twice for references).
When importing into the thesis, adjust the `\lstinputlisting` path to the
example file. The thesis controls chapter numbering and typography. Include the installation
file after `\appendix` to resolve the chapter's appendix reference.

The draft was checked against the local working tree on 16 September 2026
(base commit `974e0c472d2d7c9e3cf38a57841c72d2adbaf724`, with local changes).
It documents the implementation, rather than assuming the earlier supplied
draft's architectural claims are all implemented.

Primary implementation references:

| Chapter content | Repository source |
| --- | --- |
| Installation, dependency declarations, research status | `README.md`, `pyproject.toml` |
| Public imports | `src/kappa_planner/__init__.py` |
| Geometric objects and intermediate circles | `src/kappa_planner/geometry.py` |
| Corridor dimensions, head/tail, update and shrink | `src/kappa_planner/corridor.py` |
| Vehicle models and radius formulas | `src/kappa_planner/vehicle.py` |
| Primitive geometry, timing, sampling | `src/kappa_planner/trajectory.py` |
| Modes, dispatch, initialization, limitations and timing | `src/kappa_planner/motion_planner.py` |
| Pose-to-pose families and strict distance restriction | `src/kappa_planner/helpers/pose_to_pose_unicycle.py` |
| Additional OCP dependencies | `src/kappa_planner/helpers/ocp_pose_to_pose_unicycle.py` |

The repository URL in the chapter comes from the local README. External ROS 2
and AVP integration claims were not needed for this short scope and were not
independently verified. No performance or global-optimality claims are inferred
from function names. The declared Python minimum in package metadata is omitted
because it is not a verified compatibility matrix. No planner implementation
was changed for this draft.
