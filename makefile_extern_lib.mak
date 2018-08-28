#TIFF_DIR="C:/GnuWin32"

#IPPLIB_DIR = "C:/Intel/IPP/5.0/ia32"
#LIBINTEL="C:/Intel/Compiler/C++/10.0.026/IA32/lib"

#GDAL_DIR="C:/gdalwin32-1.4.1"
#GDAL_DIR="C:/warmerda/bld"

#CVLIB_DIR = "C:/Intel/OpenCV_1_0"
#CVLIB_DIR = "c:/Intel/OpenCV_2_3"
#CVLIB_DIR = "C:/Intel/OpenCV_2_1"
USE_MY_LIB = yes


########################################
ifdef USE_MY_LIB

source_mylib_math_dir = $(MYLIB_INCLUDE_DIR)/math

mylib_options_include =	$(opt_inc) "$(source_mylib_math_dir)"

endif
#########################################

ifdef USE_QT

ifdef DEBUG
library+=\
	$(QT_DIR_VER)\lib\qtmaind.lib $(QT_DIR_VER)\lib\QtCored4.lib $(QT_DIR_VER)\lib\QtGuid4.lib \
	$(QT_DIR_VER)\lib\Qt3Supportd4.lib $(QT_DIR_VER)\lib\QtNetworkd4.lib $(QT_DIR_VER)\lib\QtXmld4.lib
else
#	library+=\
#	$(QT_DIR_VER)\lib\qtmain.lib $(QT_DIR_VER)\lib\QtCore4.lib $(QT_DIR_VER)\lib\QtGui4.lib \
#	$(QT_DIR_VER)\lib\Qt3Support4.lib $(QT_DIR_VER)\lib\QtNetwork4.lib $(QT_DIR_VER)\lib\QtXml4.lib

	library+=\
	$(QT_DIR_VER)\lib\qtmain.lib $(QT_DIR_VER)\lib\QtCore4.lib $(QT_DIR_VER)\lib\QtGui4.lib

endif

#опиции для компилятора С++ для работы с Qt 4.x
qt_options= /D "USE_QT" /D "QT3_SUPPORT" /D "QT_DLL" /D "QT_THREAD_SUPPORT" \
	/D "QT_LARGEFILE_SUPPORT" /D "QT_EDITION=QT_EDITION_DESKTOP" /D "QT_GUI_LIB" /D "QT_CORE_LIB" 	

qt_options+= \
	$(opt_inc) "$(QT_DIR_VER)\include" \
	$(opt_inc) "$(QT_DIR_VER)\include\QtGui" \
	$(opt_inc) "$(QT_DIR_VER)\include\QtCore" \
	$(opt_inc) "$(QT_DIR_VER)\include\Qt3Support" \
	$(opt_inc) "$(QT_DIR_VER)\include\Qt"  \
	$(opt_inc) "$(QT_DIR_VER)\include\QtNetwork" \
	$(opt_inc) "$(QT_DIR_VER)\include\QtXml" 

endif

#поключаем GDAL
ifdef USE_GDAL
library+=\
			$(GDAL_DIR)/lib/gdal_i.lib

lib_options_include +=\
			$(opt_inc) "$(GDAL_DIR)/include"

cpp_options += /D "USE_GDAL"		
endif

#поключаем LIBTIFF
ifdef USE_LIBTIFF			
library+=\
			$(TIFF_DIR)/lib/libtiff.lib

lib_options_include +=\
			$(opt_inc) "$(TIFF_DIR)/include"

cpp_options += /D "USE_LIBTIFF"

endif

#поключаем IPP
ifdef USE_IPP
library+=\
			$(IPPLIB_DIR)/stublib/ippi.lib\
			$(IPPLIB_DIR)/stublib/ipps.lib\
			$(IPPLIB_DIR)/stublib/ippcore.lib\
			$(IPPLIB_DIR)/stublib/ippcv.lib

cpp_options += /D "USE_IPP"
endif

#поключаем OpenCV
ifdef USE_OPENCV
cpp_options += /D "USE_OPENCV"

ifdef USE_OPENCV_2_3
#library +=\
#			$(CVLIB_DIR)/my/lib/opencv_calib3d231.lib\
#			$(CVLIB_DIR)/my/lib/opencv_core231.lib\
#			$(CVLIB_DIR)/my/lib/opencv_highgui231.lib

library +=\
			$(CVLIB_DIR)/x86/vc9/lib/opencv_calib3d231.lib\
			$(CVLIB_DIR)/x86/vc9/lib/opencv_core231.lib\
			$(CVLIB_DIR)/x86/vc9/lib/opencv_highgui231.lib

cpp_options += /D "USE_OPENCV_2_3"
lib_options_include +=\
			$(opt_inc) "$(CVLIB_DIR)/include" \
			$(opt_inc) "$(CVLIB_DIR)/include/opencv"

else

ifdef DEBUG
library+=\
			$(CVLIB_DIR)/lib/cv210d.lib\
			$(CVLIB_DIR)/lib/cvaux210d.lib\
			$(CVLIB_DIR)/lib/cxcore210d.lib\
			$(CVLIB_DIR)/lib/highgui210d.lib
else
library+=\
			$(CVLIB_DIR)/lib/cv210.lib\
			$(CVLIB_DIR)/lib/cvaux210.lib\
			$(CVLIB_DIR)/lib/cxcore210.lib\
			$(CVLIB_DIR)/lib/highgui210.lib
endif

lib_options_include +=\
			$(opt_inc) "$(CVLIB_DIR)/include/opencv"
endif			

endif			

#поключаем MKL
ifdef USE_MKL
library+=\
			$(LIB_MKL)/ia32/lib/mkl_c.lib $(LIB_MKL)/ia32/lib/mkl_s.lib \
			$(LIB_MKL)/ia32/lib/mkl_lapack.lib $(LIB_MKL)/ia32/lib/mkl_ia32.lib \
			$(LIB_MKL)/ia32/lib/mkl_solver.lib

cpp_options += /D "USE_MKL"

lib_options_include +=\
			$(opt_inc) "$(LIB_MKL)/include"


library+= \
			$(LIBINTEL)/libmmt.lib \
			$(LIBINTEL)/libircmt.lib \
			$(LIBINTEL)/libguide.lib \
			$(LIBINTEL)/libirc.lib
endif			


# установки для cuda
ifdef USE_CUDA

#cpp_options+= $(opt_inc) "$(CUDA_DIR)/include" -D"USE_CUDA"
cpp_options+= $(opt_inc) "$(CUDA_DIR)/include" /D "USE_CUDA"
library+= $(CUDA_DIR)/lib/Win32/cuda.lib $(CUDA_DIR)/lib/Win32/cudart.lib

cuda_options=-I"$(CUDA_DIR)/include" -D"USE_CUDA"
cuda_options+= -I"$(source_mylib_common_dir)" -I"$(source_mylib_common_gpu_dir)"

#если используется внешняя библиотека
ifdef USE_OPENCV
cuda_options+= -D"USE_OPENCV"

ifdef USE_OPENCV_2_3
cuda_options+= -D"USE_OPENCV_2_3"
cuda_options+= -I"$(CVLIB_DIR)/include" -I"$(CVLIB_DIR)/include/opencv"
else
cuda_options+= -I"$(CVLIB_DIR)/include/opencv"
endif

endif

command_comp_cuda=$(CUDA_DIR)/bin/nvcc.exe  -m32 -arch sm_20 -c -ccbin=$(CC_BIN_DIR) $(cuda_options_include) $(cuda_options_mylib) $(cuda_options) -o=$@ $?

endif

