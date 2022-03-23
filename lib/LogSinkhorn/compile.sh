#!/bin/bash

CC="gcc -std=c++11"

##############################

# adjust paths in following variables:

# location of python binaries / executables
PYTHON_PATH="/opt/miniconda3/bin/"

# location of pybind11
PYBIND_PATH="/media/daten/Research/Code/OTToolbox/OTToolbox_v0.2.0/src/pybind11/include/"

##############################
# extract python stuff from python-config
PYTHON_EXTENSION_SUFFIX="`${PYTHON_PATH}python3-config --extension-suffix`"
PYTHON_INCLUDES="`${PYTHON_PATH}python3-config --includes`"


# assemble aux variables for compilation
FLAGS_INCLUDE="-I${PYBIND_PATH} ${PYTHON_INCLUDES}"
FLAGS_MISC="-fPIC -O3 -Wall -Wextra"
FLAGS_LINKER="-lm -lstdc++"



#########################

$CC $FLAGS_INCLUDE $FLAGS_MISC -c\
	LogSinkhorn.cpp -o LogSinkhorn.o


$CC -shared -o LogSinkhorn${PYTHON_EXTENSION_SUFFIX} \
	LogSinkhorn.o \
	$FLAGS_LINKER


