rem set CC_BIN_DIR=C:/VS2008/VC/bin
set CC_BIN_DIR=C:\MSVS\2010\VC\bin
call %CC_BIN_DIR%/VCVARS32.BAT

rem set QT_DIR_VER=C:/Qt/4.7.1
rem set QT_DIR_VER=C:/qt/4.7.1
set QT_DIR_VER=C:/qt/4.8.5
call %QT_DIR_VER%/bin/qtvars.bat

rem set CUDA_DIR=C:/CUDA/tkt4_32/v4.0

rem set path=%path%;%CUDA_DIR%/open64/bin;%CUDA_DIR%/bin;%CUDA_DIR%/open64/lib;
echo %path%

set BASE_DIR=%CD%

mingw32-make.exe -f makefile.mak all -e MAKE_FOR_SYSTEM=make_win.mak %1
rem mingw32-make.exe  -f  makefile.mak program -e MAKE_FOR_SYSTEM=make_win.mak %1
