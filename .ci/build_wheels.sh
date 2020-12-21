#!/usr/bin/env bash

set -ex

echo "Building for ${PYBIN}..."

# Install cython so that the files will be re-cythonized, to account for old numpy version
${PYBIN}/pip install --user cython

# Numpy must be available for openTSNE to be built
${PYBIN}/pip install --user numpy==1.16.6

# List installed dependency versions
${PYBIN}/pip freeze

# Force wheel to use old version of numpy, otherwise it tries to download latest version
echo numpy==1.16.6 > requirements_numpy.txt
# Compile openTSNE wheels
${PYBIN}/pip wheel -w wheelhouse/ -r requirements_numpy.txt .

# Bundle external shared libraries into the wheels
for whl in wheelhouse/openTSNE*.whl; do
    auditwheel repair "$whl" --plat $PLAT -w wheelhouse/
done

# Make sure the wheel can be installed
${PYBIN}/pip install --user --force-reinstall --find-links wheelhouse openTSNE
