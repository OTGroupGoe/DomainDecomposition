# Install script for directory: /content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/src/Common

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../bin/libCommon.a")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../bin" TYPE STATIC_LIBRARY FILES "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/Common/libCommon.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../include/Common/ErrorCodes.h;/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../include/Common/GridTools.h;/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../include/Common/MultiScaleTools.h;/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../include/Common/TCostFunctionProvider.h;/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../include/Common/TCouplingHandler.h;/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../include/Common/TEpsScaling.h;/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../include/Common/THierarchicalCostFunctionProvider.h;/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../include/Common/THierarchicalPartition.h;/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../include/Common/THierarchyBuilder.h;/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../include/Common/Tools.h;/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../include/Common/TVarListHandler.h;/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../include/Common/Verbose.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../include/Common" TYPE FILE FILES
    "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/src/Common/ErrorCodes.h"
    "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/src/Common/GridTools.h"
    "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/src/Common/MultiScaleTools.h"
    "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/src/Common/TCostFunctionProvider.h"
    "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/src/Common/TCouplingHandler.h"
    "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/src/Common/TEpsScaling.h"
    "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/src/Common/THierarchicalCostFunctionProvider.h"
    "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/src/Common/THierarchicalPartition.h"
    "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/src/Common/THierarchyBuilder.h"
    "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/src/Common/Tools.h"
    "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/src/Common/TVarListHandler.h"
    "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/src/Common/Verbose.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/Common/Models/cmake_install.cmake")

endif()

