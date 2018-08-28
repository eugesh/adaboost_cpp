#ifndef __DOP_FUNC_H__
#define __DOP_FUNC_H__

int sign(double x) {
   int s;
   s = (x > 0.0) ? 1 : -1;
   return s;
}

int sign(int x) {
   int s;
   s = (x > 0.0) ? 1 : -1;
   return s;
}

#endif