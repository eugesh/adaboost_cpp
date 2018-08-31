include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

link_directories ("${CMAKE_CURRENT_SOURCE_DIR}/../bin")

# set (HEADERS
#	main_window.h)

set(CMAKE_AUTORCC ON)

# QT4_WRAP_UI(UISrcs ${CMAKE_CURRENT_SOURCE_DIR}/../form_ui/mainwindow.ui)

# QT4_WRAP_CPP(MOCSrcs HEADERS)
# QT4_WRAP_CPP(MOCSrcs main_window.h)

add_executable (adaboost
	            main.cpp
                adaboost_cumsum_lib.cpp
  	            ${MOCSrcs} ${UISrcs})

if(CMAKE_VERSION VERSION_LESS "3.0")
	target_link_libraries (adaboost ${ADDON_LIBRARIES} ${QTLIBS})
else()
	target_link_libraries (adaboost ${ADDON_LIBRARIES} ${QTLIBS})
endif()
