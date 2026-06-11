#!/usr/bin/env bash
# Build and install the patched pcst_fast.
#
# pcst_fast 1.0.10 (the only build-able version on modern Python) has a
# Python-binding bug that corrupts the result arrays: the C++ binding
# declares its return type as `py::array_t<int>` but writes `int64_t`
# values into the buffer.  pybind11 then returns an `int32` numpy array
# whose contents are meaningless (always [0, 0, ..., 0] for vertices, and
# garbage for edges).  See `algorithms/pcst.py` for the symptom.
#
# We patch one line in `src/pcst_fast_pybind.cc` (changing the return
# type from `int` to `int64_t`) and rebuild.  The fix is local, builds
# in seconds, and produces correct results matching the original
# Goemans-Williamson algorithm.

set -e

WORK_DIR="${TMPDIR:-/tmp}/pcst_fast_build_$$"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "[install_pcst_fast] cloning fraenkel-lab/pcst_fast ..."
git clone --depth 1 https://github.com/fraenkel-lab/pcst_fast.git
cd pcst_fast

echo "[install_pcst_fast] patching src/pcst_fast_pybind.cc ..."
# Change the declared return type from `py::array_t<int>` to
# `py::array_t<int64_t>` so it matches the actual int64 element write.
sed -i 's/py::array_t<int>, py::array_t<int>/py::array_t<int64_t>, py::array_t<int64_t>/' \
    src/pcst_fast_pybind.cc

# Verify the patch took
grep -q 'py::array_t<int64_t>, py::array_t<int64_t>' src/pcst_fast_pybind.cc \
  || { echo "[install_pcst_fast] PATCH FAILED" >&2; exit 1; }

echo "[install_pcst_fast] building & installing ..."
# `--break-system-packages` is required on Debian/Ubuntu Python 3.12+
PIP_FLAGS="--force-reinstall --no-deps"
if [ "$(python3 -c 'import sys; print(sys.version_info >= (3, 12))')" = "True" ]; then
  PIP_FLAGS="$PIP_FLAGS --break-system-packages"
fi
python3 -m pip install . $PIP_FLAGS

echo "[install_pcst_fast] verifying ..."
python3 - <<'PY'
import numpy as np
import pcst_fast

edges = np.array([[0,1],[1,2]], dtype=np.int64)
prizes = np.array([10.0, 0.0, 10.0])
costs = np.array([1.0, 1.0])
nodes, edges_out = pcst_fast.pcst_fast(edges, prizes, costs, -1, 1, 'strong', 0)
assert nodes.dtype == np.int64, f'expected int64 nodes, got {nodes.dtype}'
assert sorted(nodes.tolist()) == [0, 1, 2], f'expected [0,1,2], got {sorted(nodes.tolist())}'
assert sorted(edges_out.tolist()) == [0, 1], f'expected [0,1], got {sorted(edges_out.tolist())}'
print('  pcst_fast verified: nodes/edges return correct int64 indices.')
PY

echo "[install_pcst_fast] done."
cd /
rm -rf "$WORK_DIR"
