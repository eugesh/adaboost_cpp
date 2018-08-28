#ifndef __COMMON_OS_H__
#define __COMMON_OS_H__

#include "common_c.h"

//#ifdef USE_QT
#if 1
//#include <QApplication>
#include <QVector>
#include <QString>
#include <QPair>
#include <utility>
#else
#include <vector>
#include <string>
#include <utility>
#define QVector std::vector
#define QString std::string
#define QPair std::pair
#endif

#ifdef WIN32 //(defined WIN32 //|| defined WIN64)
#include <windows.h>
#define __RND_SEED (GetTickCount())
#else
#define __RND_SEED (1000) //!< заменить на системный вызов ОС либо какой-нибудь остаток float(он случаен)
#endif

/*
// для создания DLL
#if (defined WIN32 || defined WIN64)
#ifdef USE_ALIB_EXP
#define ALIB_EXP __declspec(dllexport)
#else
#define ALIB_EXP 
//__declspec(dllimport)
#endif
#else
#define ALIB_EXP
#endif  
*/
#endif
