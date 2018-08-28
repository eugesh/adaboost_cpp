#include <iostream>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include "adaboostlib.h"

int main( int argc, char* argv[] ) {
   CAdaBoostThresholdClassifierCumSum abtccs ;
   CAdaBoostThresholdClassifier abctc ;

   QVector < QVector < double > > qvdFirst ;
   QVector < QVector < double > > qvdSecond ;
   qvdFirst.resize(3) ;
   qvdSecond.resize(3) ;

   /*8qvdFirst[0] << 1 << 1 ;
   qvdFirst[1] << 2 << 4 ;
   qvdFirst[2] << 5 << 3 ;
   qvdSecond[0] << 1 << 5 ;
   qvdSecond[1] << 3 << 2 ;
   qvdSecond[2] << 4 << 5 ;*/

   qvdSecond[0] << 1 << 1 ;
   qvdSecond[1] << 2 << 4 ;
   qvdSecond[2] << 5 << 3 ;
   qvdFirst[0] << 1 << 5 ;
   qvdFirst[1] << 3 << 2 ;
   qvdFirst[2] << 4 << 5 ;
   
   QVector < QVector < double > > temp ;
   temp = qvdFirst + qvdSecond ;

   std::cout << temp[3][1] << std::endl ;

   abtccs.setNumOfIter( 5 ) ;
   std::cout << "Init ok" << std::endl ;
   abtccs.training(qvdFirst, qvdSecond) ;

   return 0;
}

