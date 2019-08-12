#!/usr/bin/env bash

set -ex

echo "Building for ${PYBIN}..."

# Numpy must be available for openTSNE to be built
${PYBIN}/pip install --user numpy

# Compile openTSNE wheels
${PYBIN}/pip wheel -w wheelhouse/ .

# Bundle external shared libraries into the wheels
for whl in wheelhouse/openTSNE*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w wheelhouse/
done

# Make sure the wheel can be installed
${PYBIN}/pip install --user --force-reinstall --find-links wheelhouse openTSNE
