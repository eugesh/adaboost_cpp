############################################################
# Executable modules and extern libraries for win32
############################################################

CPP=$(CC_BIN_DIR)\cl
LINK=link
MAKE=mingw32-make.exe
MOC=$(QT_DIR_VER)\bin\moc.exe
UIC=$(QT_DIR_VER)\bin\uic.exe

TIFF_DIR="C:/GnuWin32"
#IPPLIB_DIR = "C:/Intel/IPP/5.0/ia32"

GDAL_DIR="C:/gdalwin32-1.4.1"
#GDAL_DIR="C:/warmerda/bld"

#CVLIB_DIR = "C:/Intel/OpenCV_1_0"
#CVLIB_DIR = "c:/Intel/OpenCV_2_3"
#CVLIB_DIR = "C:/Intel/OpenCV_2_1"

# Own library
MYLIB_INCLUDE_DIR = E:/Shtanov/projects/my_lib/include
############################################################

# switched flags
#DEBUG=yes
#USE_IPP=yes
#USE_MKL=yes
USE_QT=yes
#USE_CUDA=yes
#USE_GDAL=yes
#USE_LIBTIFF=yes
#USE_OPENCV=yes
#USE_OPENCV_2_3=yes

############################################################
# flags
############################################################
# расширение для объектных файлов
ext_obj=obj

# команда удаления файлов
command_del= rm -f

SD=\\

# знак для задания имени файла в include make
opt_f_b=<
opt_f_e=>

# опция для задания директорий с заголовочными файлами
opt_inc=/I
# опиция f для make
opt_make_f=-f

############################################################
# main windows libraries
############################################################

# опиции для компилятора С++  /RTC1 /Qopenmp
ifdef DEBUG
cpp_options=  /EHs /GR /D "WIN32"  /D "DEBUG" /D "_CONSOLE"  /D "WIN32API"  \
				/D "_CRT_SECURE_NO_DEPRECATE"   \
				/D "__MSVC_RUNTIME_CHECKS"  \
				/Zc:wchar_t- /MDd  /Zl /Zp16 /RTC1
else
cpp_options=  /EHs /D "WIN32"   /D "_CONSOLE"  /D "WIN32API"  \
				/D "_CRT_SECURE_NO_DEPRECATE"   \
				/Zc:wchar_t- /MT /QaxNPT /Zp16 /Qparallel /Qpar-report1  /Qvec-report1 /Qopenmp
endif

cpp_options += /D "ABC_EXPORT_DLL"
############################################################
#подключение внешних библиотек
include $(base_dir)/makefile_extern_lib.mak

#command_comp=$(CPP) /c $(cpp_options) $(cpp_options_include) $(qt_options) $(cpp_options_cuda) $(mylib_options_include) /Fo$@ $?

############################################################

command_comp=$(CPP) /c $(cpp_options) $(cpp_options_include) $(lib_options_include) $(qt_options) $(cpp_options_cuda) $(mylib_options_include) /Fo$@ $?
command_touch=touch $@
command_linkDLL=$(LINK) /OUT:$@ /machine:X86 /DLL


