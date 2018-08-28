#ifndef __ABC_EXPORT_H__
#define __ABC_EXPORT_H__

#ifdef ABC_EXPORT_DLL
	#if defined( WIN32 )
		#define ABC_DLL     __declspec(dllexport)
	#else
		#define ABC_DLL
	#endif
#else// #ifdef TOUCH_EXPORT_DLL
	#if defined( WIN32 )
		#define ABC_DLL     __declspec(dllimport)
	#else
		#define ABC_DLL
	#endif
#endif// #ifdef TOUCH_EXPORT_DLL

#endif