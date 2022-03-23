#!/bin/bash

CC="gcc -std=c++11"

##############################

# adjust paths in following variables:

# location of eigen library
EIGEN_PATH="/usr/include/eigen3"

# location of python binaries / executables
PYTHON_PATH="/opt/miniconda3/bin/"

# location of OTToolbox
OTTOOLBOX_PATH="/media/daten/Research/Code/OTToolbox/OTToolbox_v0.2.0/"

# location of pybind11
PYBIND_PATH="/media/daten/Research/Code/Extern/pybind11/pybind11/include/"

##############################
# extract python stuff from python-config
PYTHON_EXTENSION_SUFFIX="`${PYTHON_PATH}python3-config --extension-suffix`"
PYTHON_INCLUDES="`${PYTHON_PATH}python3-config --includes`"


# assemble aux variables for compilation
FLAGS_INCLUDE="-I${OTTOOLBOX_PATH}include -I${EIGEN_PATH} -I${PYBIND_PATH} ${PYTHON_INCLUDES}"
FLAGS_MISC="-fPIC -O3 -Wall -Wextra"
FLAGS_LINKER="-L${OTTOOLBOX_PATH}bin/ -lSinkhorn -lCommon -lm -lstdc++"




#########################

$CC $FLAGS_INCLUDE $FLAGS_MISC -c\
	CPPSinkhorn.cpp -o CPPSinkhorn.o

$CC -shared -o CPPSinkhorn${PYTHON_EXTENSION_SUFFIX} \
	CPPSinkhorn.o \
	$FLAGS_LINKER


