set QT_DIR_VER=C:\Qt\4.7.1
set path=%path%;%QT_DIR_VER%\bin;

rem set CUDA_DIR=C:/CUDA/tkt4_32/v4.0
set path=%path%;%CUDA_DIR%/open64/bin;%CUDA_DIR%/bin;%CUDA_DIR%/open64/lib;

echo %path%
call exe\abc_class.exe