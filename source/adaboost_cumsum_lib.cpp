#include "adaboost_cumsum_lib.h"
#include "dop_func.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <QFile>
#include <QTextStream>

using namespace std;

/*
  * Auxilliary functions.
  */

bool lessThen (const TrainingSetNumerator &a1,const TrainingSetNumerator &a2) {
   return a1.point<=a2.point ;
}

double maxQV( QVector<double> array ) {
   double xMax = array[0] ;

   for(int i=0;i<array.size();++i) {
      if( xMax < array[i] ) {
         xMax = array[i];
      }
   }
   return xMax;
}

double minQV( QVector<double> array ) {
   double xMin = array[0];

   for(int i=0;i<array.size();++i) {
      if( xMin > array[i] ) {
         xMin = array[i];
      }
   }
   return xMin;
}

double maxQV( QVector<double> array , int&position) {
   double xMax = array[0];
   position = 0;

   for(int i=0;i<array.size();++i) {
      if( xMax < array[i] ) {
         xMax = array[i];
         position = i;
      }
   }
   return xMax;
}

double minQV( QVector<double> array , int&position) {
   double xMin = array[0];
   position = 0;
   
   for(int i=0;i<array.size();++i) {
      if( xMin > array[i] ) {
         xMin = array[i];
         position = i;
      }
   }
   return xMin;
}

// Abstract classes constructors and destructors (Interfaces)

ABinaryClassifier::~ABinaryClassifier () {
   cout<<"ABinaryClassifier::Dstr" << endl;
}

// Derived classes constructors and destructors  (Realisations)
CBinaryWeakThresholdClassifierCumSum::CBinaryWeakThresholdClassifierCumSum (void) 
{
   cout<<"CBinaryWeakThresholdClassifierCumSum::Cstr" << endl;
}

CBinaryWeakThresholdClassifierCumSum::~CBinaryWeakThresholdClassifierCumSum 
  (void) {
   cout<<"CBinaryWeakThresholdClassifierCumSum::Dstr" << endl;
}

CAdaBoostThresholdClassifierCumSum::CAdaBoostThresholdClassifierCumSum () 
  : iNumOfIter(1), iDescriptorSize(1) {
   cout<<"CAdaBoostThresholdClassifierCumSum::Cstr"<<endl;
}

CAdaBoostThresholdClassifierCumSum::~CAdaBoostThresholdClassifierCumSum () {
   cout<<"CAdaBoostThresholdClassifierCumSum::Dstr"<<endl;
}

int 
CBinaryWeakThresholdClassifierCumSum::testing(QVector<double> const& qvdArray)const {
   int rez ;

   if (qvdArray[abs(Dim) - 1] <= Th)
      rez =  1 * sign(Dim);
   else
      rez = -1 * sign(Dim);

   return rez ;
}

//int CBinaryWeakThresholdClassifierCumSum::testing(QVector<double> const& qvdArray)const {
  // return ( (qvdArray[abs(Dim) - 1] <= Th) ? 1 : -1 ) * sign(Dim);
//}

void CAdaBoostThresholdClassifierCumSum::recomputeWeight ( QVector<int> const& pLabels, QVector<int> const& h_m, QVector<double> & w_d, double dalpha_m, double errorm  )const {
   int iN ;
   iN = pLabels.size () ;

   //cout << " w_d soon aFTER exp: " << endl ;
   for (int i = 0; i < iN; ++i) {
      w_d[i] *= exp( -max( dalpha_m, dEps ) * h_m[i] * pLabels[i] ); // recalculation of D weights
      //cout << w_d[i] << " " ;
   }
   //cout << endl;

   double sum_w = 0.0 ;

   for ( int i = 0 ; i < iN; ++i )
      sum_w += w_d[i] ; // sum of w_d
      
   for ( int i = 0; i < iN; ++i ) {
      w_d[i] /= max( sum_w, dEps ) ; // sum of weights should be equal to 1 , reweighting
   }

}

double CAdaBoostThresholdClassifierCumSum::dComputeError ( QVector<int> const& pLabels, QVector<int> const& h_m, QVector<double> const& qvdWeight )const { // returns computed error
   double error_m = 0;

   for(int i=0;i < pLabels.size ();++i)
      error_m += abs( qvdWeight[i] * ( h_m[i] != pLabels[i] ) ) ;  // Compute error of current weak clissifier h_func_m

   return error_m ;
}

double CAdaBoostThresholdClassifierCumSum::dCalcAlpha_m ( double error_m ) const {
   return ( 0.5 ) * log ( ( 1 - error_m ) / max( error_m, dEps ) ) ;
}

int CAdaBoostThresholdClassifierCumSum::training( QVector<tDescriptor> const& qvdFirst, QVector<tDescriptor> const& qvdSecond ) {
   if( qvdFirst.isEmpty() || qvdSecond.isEmpty() ) {
      std::cout << "Error: empty training data." ;
      return -1;
   }
   if( qvdFirst[0].size() != qvdSecond[0].size() ) {
      std::cout << "Error: Incorrect training data." ;
      return -1;
   }

   iDescriptorSize = qvdFirst[0].size() ;

   // The training set
   QVector<QVector<double> > qvqvdTrainSet = qvdFirst + qvdSecond ;
   int iNumOfElements = qvqvdTrainSet.size() ; // qvdFirst.size () + qvdSecond.size () ;
   
   // The indexes of each element in the trainig set after sorting
   QVector<QVector<int> >  qvqviIndexes ;

   // Set a size for the estimation array. Estimations are accumulated on each training iteration
   qvdEstimation.resize(iNumOfElements) ;

   qvqviIndexes.resize(iNumOfElements) ;

   // _Preparations_
   // Sorting
   qvqviIndexes = qvqviSortByEachDim( qvqvdTrainSet ) ;
   // Labeling
   QVector<int> qvIntLabels = qviSeparate( qvdFirst.size(), iNumOfElements, 1, -1 ) ;

   QVector<double> qvdCumSum ;    // cumulative sums
   QVector<double> qvdWeights ;   // weights of each element
   QVector<int> qviH_m ;  // weak classifier

   qvdCumSum.resize(iNumOfElements) ;
   qviH_m.resize(iNumOfElements);

   for ( int i = 0; i < iNumOfElements ; ++i ) {
      qvdWeights << 1.0 / iNumOfElements ;
   }
   qvWeakClassifier.resize(iNumOfIter);

   // Adaboost iteration cycle
   for ( int m = 0; m < iNumOfIter; ++m ) { 
       std::cout << "_____________________________________" << std::endl ;
       std::cout << "No error on " << m << " iteration." << std::endl ;
       std::cout << "_____________________________________" << std::endl ;
       {  // weak classifier training 
          qvWeakClassifier[m].setWeights(qvdWeights) ;
          qvWeakClassifier[m].setSortedIndexes(qvqviIndexes) ;
 
          qvWeakClassifier[m].training(qvdFirst, qvdSecond) ;
          std::cout << "Weak classifier is trained successfully!" << std::endl ;
       }

       for(int j = 0; j < iNumOfElements; ++j)
          qviH_m[j] = qvWeakClassifier[m].testing(qvqvdTrainSet[j]) ;

       double dError_m = dComputeError(qvIntLabels, qviH_m, qvdWeights) ;
       cout << "dError_m = " << dError_m << endl ;

       if (abs(dError_m) > 0.5) {
          cout << "Warning: abs(dError_m[m]) > 0.5!!! Stop training. " << endl ;  // if current error more than 50% -> stop training
       }

       qvdLm << dCalcAlpha_m(dError_m) ;
       cout << "qvdLm = " << qvdLm.last() << endl ;

       recomputeWeight(qvIntLabels, qviH_m, qvdWeights, qvdLm.last(), dError_m) ;

       computeEstimations(qviH_m, qvIntLabels, qvdFirst.size() );

   }

   return 0;
}

void CAdaBoostThresholdClassifierCumSum::computeEstimations (QVector<int> const& h_m,QVector<int> const& qvIntLabels , int iNofFirstSet ) {
     int iN ;
     iN = h_m.size () ;
     qvdFirstOut.resize(iNofFirstSet) ;
     qvdFirstOutScore.resize(iNofFirstSet) ;
     qvdSecondOut.resize(iN-iNofFirstSet) ;
     qvdSecondOutScore.resize(iN-iNofFirstSet) ;
     
     qvdTotalErrors << 0;  // push new element
     
      //________Estimation____________
      for (int i = 0; i < iN; ++i) {
         qvdEstimation[i] += qvdLm.last() * h_m[i]; // * sign(pOutStumps[m].Dim);  // calculate estimation of current rate of each element
         if( i < iNofFirstSet  ) {
            qvdFirstOut[i] = sign(qvdEstimation[i]);    // define class of current element
            qvdFirstOutScore[i] = qvdEstimation[i];
         }
         
         else {
            qvdSecondOut[ i - iNofFirstSet ] = sign(qvdEstimation[i]) ;
            qvdSecondOutScore[ i - iNofFirstSet ] = qvdEstimation[i] ;
         }

         if ( sign(qvdEstimation[i]) != qvIntLabels[i] ) {
            qvdTotalErrors.last() += (double) 1 / (double) iN ;
         }
      }
}

int CAdaBoostThresholdClassifierCumSum::testing(tDescriptor const& qvdArray)const {

   double estimateclasssum_te = 0;

   for (int m = 0; m < qvWeakClassifier.size() ; ++m ) {// num of iteration cycle

      //weak classifier testing
      int h = qvWeakClassifier[m].testing(qvdArray);
      //printf("h = %d\n",h);

      //estimation for all trained weak classifiers
      estimateclasssum_te = estimateclasssum_te + qvdLm[m] * h ; // * sign( qvWeakClassifier[m].iGetDim() );
   }

   return sign(estimateclasssum_te) ;
}

QVector<int> CAdaBoostThresholdClassifierCumSum::testing(QVector<tDescriptor> const& desc)const {
   QVector<int> qviLabels ;

   for(int i = 0; i < desc.size(); ++i)
      qviLabels << testing(desc[i]);

   return qviLabels ;
}

/**
return indexes of sorted aray
*/
QVector<QVector<int> > CAdaBoostThresholdClassifierCumSum::qvqviSortByEachDim(QVector<tDescriptor> const& qvqvdCoords ) {
   if(qvqvdCoords.isEmpty()) {
      cout << "CAdaBoostThresholdClassifierCumSum::qvqviSortByEachDim : empty input data" << endl;
      return QVector<QVector<int> >();
   }

   QVector<QVector<int> > qvqviIndexes ;

   int iN ;
   iN = qvqvdCoords.size () ;
   int iDescrSize = qvqvdCoords.first().size();

   QVector<TrainingSetNumerator> list ; // the vector of each element on one dimension
   qvqviIndexes.resize(iN) ;

   for ( int d = 0; d < iDescrSize; ++d ) {  // the dimension cycle  // this the private var iDescriptorSize is being used
      list.clear() ;
      for ( int i = 0; i < iN; ++i ) {   // the cycle for all elements
         TrainingSetNumerator buf( i, qvqvdCoords[i][d] ) ;
         list.push_back(buf);  // positive
      }

      sort(list.begin(), list.end(), lessThen); // sort each element of training data by each dimension

      for (int i = 0; i < iN; ++i) {
         qvqviIndexes[i] << list[i].number;  // indexes in original array
      }

   }

   return  qvqviIndexes ;
}

/** separate N elements into iFirstN and N-iFirstN sets of elements
    firstValue and secondValue - values of labels (1/-1 e.g.)
    iFirstN - num of elements in the first set
    size - num of elements 

    returns labeled array
*/
QVector<int> CAdaBoostThresholdClassifierCumSum::qviSeparate(int iFirstN, int size, int firstValue, int secondValue) {
   QVector<int> qvIntLabels ;
   
   for (int i = 0; i < size ; ++i ) {
      if ( i < iFirstN )
         qvIntLabels << firstValue ;  // separation of training set (labeling)
      else
         qvIntLabels << secondValue ;
   }

   return qvIntLabels ;
}

int CBinaryWeakThresholdClassifierCumSum::training( QVector<QVector<double> > const & qvdFirst, QVector<QVector<double> > const & qvdSecond ) {
   int iNumOfElements = qvdWeights.size () ;      

   QVector<double> qvdCumSum ;  // cumulative sums
   qvdCumSum.resize(iNumOfElements - 1) ;
   double dMinError = 0 ;

   // Vars for the best decision on m iteration
   int iDimension_m = 0 ;
   int iIndexOfMaxElement_m;
   int iIndexOfMinElement_m;
   double dMaxElement_m;
   double dMinElement_m;

   QVector<int> qviLabels ;

   QVector<QVector<double> > qvqvdTrainSet ;
   qvqvdTrainSet = qvdFirst + qvdSecond ;  // training set
   
   int iDescriptorSize = qvdFirst[0].size() ;

   for(int i = 0; i < qvqvdTrainSet.size(); ++i )
      if( i < qvdFirst.size() )
         qviLabels <<  1 ;   // labeling
      else
         qviLabels << -1 ;

   {
      // looking for extremum
      double xmax, xmin ;
      int imax, imin ;
      for ( int d = 0; d < iDescriptorSize; ++d ) { //  cycle in each dimension
          for ( int i = 0; i < iNumOfElements - 1 ; ++i ) {//cycle for each element
             qvdCumSum[i] = 0 ;  // cumulative sums init
          }

          qvdCumSum[0] = qvdWeights[ qvqvIntIndexes[0][d] ] * qviLabels[ qvqvIntIndexes[0][d] ] ;  // the first element of cumsum

   
          for (int i = 1; i < iNumOfElements - 1; ++i) {
             // cumulative sum calculation
             qvdCumSum[i] = qvdCumSum[i - 1] + ( qvdWeights[qvqvIntIndexes[i][d]] * 
                                             qviLabels[qvqvIntIndexes[i][d]] ) ;
          }
   
          /*cout << "cumsum = " << endl ;  // DEBUG
          for ( int i = 0; i < iNumOfElements - 1; ++i ) {  // this the private var iNumOfElements is being used
             cout << qvdCumSum [i] << " " ;  // cumulative sums init
          }*/
          cout << endl ;
          // find max/min element of cum_sum and index of it => extremum => the best threshold
          xmax=maxQV(qvdCumSum,imax) ;   
          xmin=minQV(qvdCumSum,imin) ;
    
          if( dMinError < max( abs(xmax), abs(xmin) ) ) {
              dMinError = max( abs(xmax), abs(xmin) ) ;
              iDimension_m = d ;  // fixate the best dimension and coordinate of the best threshold
              iIndexOfMaxElement_m = imax ;  // the index of the best threshold
              iIndexOfMinElement_m = imin ;
              dMaxElement_m = xmax ;  // the value of the best threshold
              dMinElement_m = xmin ;
          }
      }  
   }
   cout << "No error after looking for extremum" << endl ;
   {
      // Threshold selection

      Dim = iDimension_m + 1 ;
      
      cout << "Dim = " << Dim << endl ;
      cout << "iIndexOfMaxElement_m = " << iIndexOfMaxElement_m << endl ;
      cout << "iIndexOfMinElement_m = " << iIndexOfMinElement_m << endl ;
      cout << "dMaxElement_m = " << dMaxElement_m << endl ;
      cout << "dMinElement_m = " << dMinElement_m << endl ;
      cout << "qvqvIntIndexes[iIndexOfMaxElement_m][iDimension_m] = " << qvqvIntIndexes[iIndexOfMaxElement_m][iDimension_m] << endl ;
      cout << "qvqvIntIndexes[iIndexOfMinElement_m][iDimension_m] = " << qvqvIntIndexes[iIndexOfMinElement_m][iDimension_m] << endl ;
      if (abs(dMaxElement_m) > abs(dMinElement_m)) {  // determine the sign of unequality
   
         Th = qvqvdTrainSet[ qvqvIntIndexes[iIndexOfMaxElement_m][iDimension_m] ] [iDimension_m];
   
         Dim *= 1;   // forward
      }
      else {
   
         Th = qvqvdTrainSet[ qvqvIntIndexes[iIndexOfMinElement_m][iDimension_m] ] [iDimension_m];
   
         Dim *= -1;   // backward
      }
   }
   cout << "Th = " << Th << endl ;
   cout << "Dim = " << Dim << endl ;
   cout << "No error after threshold selection" << endl ;

   return 0 ;
}

int CAdaBoostThresholdClassifierCumSum::read(QString const& path) { //!< Reads parameters. Returns 
     QFile file(path);    
     if ( !file.open(QIODevice::ReadOnly) ) {
        fprintf( stdout, "CBinaryClassifier::Read : Can't read <%s>\n", qPrintable(path) ) ;
        return -1 ;
     } ;
   
     double dtemp ; // temporal variable
   
     QVector<double> alpha_vector ;
     QVector<double> th_vector ;
     QVector<int>     dim_vector ;
   
     QTextStream ts(&file);
     ts >> iNumOfIter ;
     //ts >> dtemp ;
     //ts >> dtemp ;
     ts >> iDescriptorSize ;
   
     qvWeakClassifier.clear() ;
     qvWeakClassifier.resize(iNumOfIter) ;
     qvdLm.clear() ;
     qvdLm.resize(iNumOfIter) ;
     //clear() ;
     //.resize(iNumOfIter) ;
   
     // читаем весовые коэфициенты
     for( int i = 0 ; i < iNumOfIter; i ++ )
        ts >> qvdLm[i] ;
   
     // читаем пороги
     for( int i = 0 ; i < iNumOfIter; i ++ ) {
        ts >> dtemp ;
        qvWeakClassifier[i].setThreshold(dtemp) ;
     }

     int itemp ;
      // индексы размерностей
     for( int i = 0 ; i < iNumOfIter; i ++ ) {
        ts >> itemp ;
        qvWeakClassifier[i].setDim(itemp) ;
     }
   
     // вывод результатов в консоль
     for( int i = 0 ; i < iNumOfIter; i ++ ) {
        fprintf(stdout,"%d : %lf %f %d\n", i, qvdLm[i], qvWeakClassifier[i].dGetThreshold(), qvWeakClassifier[i].iGetDim()) ;
     } ;
     //getchar();
   return 0;
}

                     /* {
                         int readTestingSet (QVector<QVector<REAL_TYPE>>&test_set,QString path, int descriptor_size) {
                         QFile file(path);
                         if (!file.open(QIODevice::ReadOnly)) {
                            printf("Error : Settings : there is not <%s>\n",path.toLocal8Bit().data()) ;
                            return -1;
                         }
                         else {
                            QString buf;
                            QVector<QStringList> qsl;
                            QTextStream ts(&file);
                            int count=0;
                            while(!ts.atEnd ( )) {
                                  ts>>buf;
                                  qsl<<buf.split(" ");
                            }
                            file.close( );
                            test_set.resize( qsl.size( ) / descriptor_size);
                            for( int i = 0; i < qsl.size( ) / descriptor_size; ++i) {
                                for( int j = 0;j<descriptor_size;++j) {
                                   test_set[i]<<qsl[count][0].toDouble();
                                   //cout<<test_set[i][j];
                                   count++;
                                }
                                //cout<<endl;
                            }
                         }
                         //cout<<test_set[0][0]<<endl;
                         //cout<<test_set[10][2]<<endl;
                         printf("No error in readTestingSet\n");
                         
                         return 0;
                      }
                     }*/

int CAdaBoostThresholdClassifierCumSum::write(QString const& path) { //!< Writes parameters. Returns 
// success(0)|fault
   FILE *fp;
   if((fp = fopen(path.toLocal8Bit().data(),"w"))==NULL) {
      printf("File %s hasn't been opened",path.toLocal8Bit().data());
      return -1;
   }
 
   fprintf(fp,"%d\n",iNumOfIter);
   fprintf(fp,"%d\n",iDescriptorSize);
   
   for(int i=0;i<qvdLm.size();++i)
      fprintf(fp,"%f ",qvdLm[i]);
   fprintf(fp,"\n");
   
   for(int i=0;i<qvWeakClassifier.size();++i)
      fprintf(fp,"%f ",qvWeakClassifier[i].dGetThreshold( ));
   fprintf(fp,"\n");
   
   for(int i=0;i<qvWeakClassifier.size();++i)
      fprintf(fp,"%d ",qvWeakClassifier[i].iGetDim( ));
   fprintf(fp,"\n");
   
   fclose(fp);

   return 0;
}
/* {

int readTestingSet (QVector<QVector<REAL_TYPE>>&test_set,QString path, int descriptor_size) {
   QFile file(path);
   if (!file.open(QIODevice::ReadOnly)) {
      printf("Error : Settings : there is not <%s>\n",path.toLocal8Bit().data()) ;
      return -1;
   }
   else {
      QString buf;
      QVector<QStringList> qsl;
      QTextStream ts(&file);
      int count=0;
      while(!ts.atEnd ( )) {
            ts>>buf;
            qsl<<buf.split(" ");
      }
      file.close( );
      test_set.resize( qsl.size( ) / descriptor_size);
      for( int i = 0; i < qsl.size( ) / descriptor_size; ++i) {
          for( int j = 0;j<descriptor_size;++j) {
             test_set[i]<<qsl[count][0].toDouble();
             //cout<<test_set[i][j];
             count++;
          }
          //cout<<endl;
      }
   }
   //cout<<test_set[0][0]<<endl;
   //cout<<test_set[10][2]<<endl;
   printf("No error in readTestingSet\n");
   
   return 0;
}
}*/
