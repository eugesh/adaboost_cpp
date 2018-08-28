#ifndef __COMMON_C_H__
#define __COMMON_C_H__

// константы
#define __PI  (3.1415926535897932384626433832795)
#define __2PI (6.283185307179586476925286766559)
#define __EPS (1.0e-16)

#define __ZERO    (0)
#define __NULL    (0)
#define __SUCCESS (0)
#define __INVALID_INDEX  (-1)
#define __INVALID_NUMBER (-2)
#define __CUSTOM_ERROR  (1001)

// минимально допустимы коэффициент коррел€ции
#define __MIN_CC_TH (0.0f)

//преобразование градусы/радианы
#define __DEG2RAD(a)  (a*__PI/180.0f)
#define __RAD2DEG(a)  (a*180.0f/__PI)

// каналы
#define __CHGRAY  (0)
#define __CHRED   (1)
#define __CHGREEN (2)
#define __CHBLUE  (3)

// извлечение цвета
#define __RED(RGB32)    ((RGB32>>16)&0xFF)
#define __GREEN(RGB32)  ((RGB32>>8)&0xFF)
#define __BLUE(RGB32)   ((RGB32&0xFF))
#define __GRAY(RGB32)   (( float(__RED   (RGB32))*11.0f + \
                           float(__GREEN (RGB32))*16.0f + \
                           float(__BLUE  (RGB32))*5.0f ) / 32.0f )

// сравнение двух величин
#define __MAX(a,b)    ((a) > (b) ? (a) : (b))
#define __CMP(a,b)    (((a) > (b)) - ((a) < (b)))
#define __SIGN(a)     __CMP((a),0)

#endif
