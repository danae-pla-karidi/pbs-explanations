"""
pcst_fast compatibility loader.

Imports `pcst_fast` and runs a one-time self-check at module load to
detect the int32-vs-int64 binding bug in the unpatched 1.0.10 release.

If the bug is detected, raises a clear `ImportError` directing the user
to `scripts/install_pcst_fast.sh`.
"""

from __future__ import annotations

import numpy as np


def _self_check() -> None:
    """Run a 3-node sanity check and raise if pcst_fast is broken.

    The unpatched 1.0.10 release returns int32 arrays whose contents
    are corrupted (always [0, 0, ..., 0]).  The patched version returns
    int64 arrays with valid integer indices.
    """
    import pcst_fast  # type: ignore

    edges = np.array([[0, 1], [1, 2]], dtype=np.int64)
    prizes = np.array([10.0, 0.0, 10.0])
    costs = np.array([1.0, 1.0])
    nodes, edges_out = pcst_fast.pcst_fast(edges, prizes, costs, -1, 1, "strong", 0)

    if nodes.dtype != np.int64:
        raise ImportError(
            f"The installed pcst_fast returns dtype={nodes.dtype}, expected int64.\n"
            "This is the broken upstream 1.0.10 binding.  Run\n"
            "    bash scripts/install_pcst_fast.sh\n"
            "to build the patched version (one-line C++ binding fix)."
        )

    expected_nodes = sorted([0, 1, 2])
    if sorted(nodes.tolist()) != expected_nodes:
        raise ImportError(
            f"pcst_fast self-check failed: got nodes={nodes.tolist()}, expected {expected_nodes}.\n"
            "The C++ algorithm is producing degenerate results.  Verify the install."
        )


_self_check()


def pcst_fast_solve(edges: np.ndarray, prizes: np.ndarray, costs: np.ndarray,
                    *, root: int = -1, num_clusters: int = 1,
                    pruning: str = "strong",
                    verbosity: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Thin wrapper around pcst_fast.pcst_fast with keyword arguments.

    Returns (node_ids, edge_ids), both int64 arrays of indices into the
    input vertex / edge lists.
    """
    import pcst_fast  # type: ignore
    return pcst_fast.pcst_fast(edges, prizes, costs,
                                int(root), int(num_clusters), pruning, int(verbosity))
