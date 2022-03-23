# Install script for directory: /content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/src/ShortCutSolver

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
   "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../bin/libShortCutSolver.a")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../bin" TYPE STATIC_LIBRARY FILES "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/ShortCutSolver/libShortCutSolver.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../include/ShortCutSolver/Interfaces.h;/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../include/ShortCutSolver/MultiScaleSolver.h;/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../include/ShortCutSolver/TShieldGenerator.h;/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../include/ShortCutSolver/TShieldGenerator_Models.h;/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../include/ShortCutSolver/TShortCutSolver.h")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/build/../include/ShortCutSolver" TYPE FILE FILES
    "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/src/ShortCutSolver/Interfaces.h"
    "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/src/ShortCutSolver/MultiScaleSolver.h"
    "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/src/ShortCutSolver/TShieldGenerator.h"
    "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/src/ShortCutSolver/TShieldGenerator_Models.h"
    "/content/gdrive/MyDrive/Colab Notebooks/DomDecKeOps/lib/MultiScaleOT/src/ShortCutSolver/TShortCutSolver.h"
    )
endif()

