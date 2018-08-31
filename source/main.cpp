#include <iostream>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include "adaboost_cumsum_lib.h"


int main( int argc, char* argv[] ) {
   CAdaBoostThresholdClassifierCumSum abtccs;

   QVector <QVector <double> > qvdFirst;
   QVector <QVector <double> > qvdSecond;
   qvdFirst.resize(3);
   qvdSecond.resize(3);

   qvdSecond[0] << 1.1 << 1.1;
   qvdSecond[1] << 2.1 << 4;
   qvdSecond[2] << 5.3 << 3;
   qvdFirst[0] << 1.3 << 5.1;
   qvdFirst[1] << 3 << 2;
   qvdFirst[2] << 4 << 5.2;

   // Set number of weak classifiers.
   abtccs.setNumOfIter(15);
   std::cout << "Init ok" << std::endl;
   abtccs.training(qvdFirst, qvdSecond);

   QVector <QVector <double> > qvd_test;
   qvd_test.resize(6);

   qvd_test[0] << 3 << 1;
   qvd_test[1] << 2 << 4;
   qvd_test[2] << 5 << 3;
   qvd_test[3] << 1 << 5;
   qvd_test[4] << 3 << 2;
   qvd_test[5] << 4 << 5;

   QVector<int> res = abtccs.testing(qvd_test);

   for(int i = 0; i < res.size(); ++i) {
	   std::cout << res[i] << std::endl;
   }

   abtccs.write("model.src");

   CAdaBoostThresholdClassifierCumSum abtccs2;

   abtccs2.read("model.src");

   res = abtccs2.testing(qvd_test);

   for(int i = 0; i < res.size(); ++i) {
	   std::cout << res[i] << std::endl;
   }

   return 0;
}
