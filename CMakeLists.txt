
cmake_minimum_required (VERSION 2.8.11)

project (scanline_viewer CXX)

include (FindPkgConfig)

if (NOT CMAKE_BUILD_TYPE)
  set (CMAKE_BUILD_TYPE "Release")
endif ()

if (${CMAKE_CXX_COMPILER_ID} STREQUAL GNU OR ${CMAKE_CXX_COMPILER_ID} STREQUAL Clang)
  set (CXX_FLAGS "-std=c++11 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers $ENV{CXXFLAGS}")
  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2 ${CXX_FLAGS}")
  set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O1 ${CXX_FLAGS}")
  set (MATH_LIBRARIES m)
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
  set (CXX_FLAGS "/std:c++latest /wd4244 /wd4305 $ENV{CXXFLAGS}")
  set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CXX_FLAGS}")
  set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CXX_FLAGS}")
else ()
  message (FATAL_ERROR "Unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
endif ()

if (NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
  set (CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin")
  set (CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin")
endif ()

if (NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
  set (CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin")
  set (CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin")
endif ()

if (NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
  set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/bin")
  set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/bin")
endif ()

message("CMAKE_CURRENT_SOURCE_DIR = " ${CMAKE_CURRENT_SOURCE_DIR})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/source")

#find_package( OpenCV REQUIRED )

#IF(NOT OpenCV_FOUND)
#    MESSAGE(FATAL_ERROR "OpenCV not found")
#ENDIF()
#set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_videoio opencv_contrib opencv_calib3d opencv_features2d opencv_gpu opencv_flann opencv_photo)

#include_directories( ${OpenCV_INCLUDE_DIRS} )

#if(CMAKE_VERSION VERSION_LESS "2.8.11")
#    include_directories(${OpenCV_INCLUDE_DIRS})
#endif()

set (QTLIBS QtGui QtCore QtNetwork)
find_package (Qt4 REQUIRED)
# qt4_wrap_cpp (MOC_SOURCES ${HEADERS})
include(${QT_USE_FILE})

#include (findeigen3)
find_package (Eigen3 REQUIRED)
include_directories (${EIGEN3_INCLUDE_DIR})
pkg_search_module(Eigen3 REQUIRED eigen3)

link_directories ("${CMAKE_CURRENT_SOURCE_DIR}/../bin")
set (ADDON_LIBRARIES ${ADDON_LIBRARIES})

add_definitions (-DG_LOG_DOMAIN="ScanlineViewer")
add_subdirectory ("source")
add_subdirectory ("custom_plot")
add_subdirectory ("tests")