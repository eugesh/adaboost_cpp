#ifndef __ADABOOSTLIB_CUMSUM_H__
#define __ADABOOSTLIB_CUMSUM_H__
#include "abc_export.h"
#include <QVector>
#include <QString>

//----------------------CBinaryClassifier-------------------
static const double dEps = 1e-6 ; // constant for comparison with zero to predict division by zero

typedef QVector<double> tDescriptor ;
/*
 Internal structure. Is used for sorting training/testing set by coordinates
*/
struct TrainingSetNumerator {  // use for sorting 
    int number; // порядковый номер элемента выборки в исходном массиве // number of element in Training Set
    double point;  // the value of current coordinate in Training Set
    TrainingSetNumerator(){;};
    TrainingSetNumerator(int n, double p) : number(n), point(p) {
    }
};
// Sort by coordinate. Internal function
bool lessThen (const TrainingSetNumerator &a1, const TrainingSetNumerator &a2) ;

/**
The binary classifier class
*/
class ABC_DLL ABinaryClassifier {  // The base class
  public: ;
   virtual ~ABinaryClassifier () ;
   
   //! метод дл€ обучени€ классификатора
   virtual int training ( QVector<tDescriptor> const& qvdFirst, QVector<tDescriptor> const& qvdSecond ) = 0;  //!< returns success(0)|fault(!0)
   
   //! метод дл€ тестировани€ классификатора
   virtual int testing  ( tDescriptor const& qvdArray )const = 0;  //!< returns rezult of testing {1,-1}
};

class ABC_DLL CBinaryWeakThresholdClassifierCumSum : public ABinaryClassifier {  // Derived class
  public: ;
   CBinaryWeakThresholdClassifierCumSum (void) ;
   virtual ~CBinaryWeakThresholdClassifierCumSum(void) ;
   
   // Initialization
   //! Set current weights
   void setWeights(QVector<double> const& qvdW) { qvdWeights = qvdW ; } ;
   //!< Set indexes of sorted training set
   void setSortedIndexes(QVector<QVector<int> > const& qvdI) { qvqvIntIndexes = qvdI ; } ;
   //! Set current weights and indexes of sorted training set
   void setParam(QVector<double> const& qvdW, QVector<QVector<int> > const& qvdI) { qvdWeights = qvdW ; qvqvIntIndexes= qvdI ; } ;

   // Classificator training function
   virtual int training (QVector<tDescriptor> const& qvdFirst, QVector<tDescriptor> const& qvdSecond) ;  //!< returns success(0)|fault(!0)

   // Testing. Classifier MUST be trained before testing
   virtual int testing (tDescriptor const& qvdArray)const ;  //!< returns rezult of testing {1,-1}

   // Functuons providing access to private data
   void setThreshold ( double t ) { Th = t ; } ;
   void setDim ( int d ) { Dim = d ; } ;
   double dGetThreshold ( )const { return Th ; } ;
   int iGetDim ( )const { return Dim ; } ;

  private: ;
   double Th ; //!< порог принятия решения
   int Dim ;   //!< размерность, по которой определяем решение, знак определяет вид неравенства - больше(+) или меньше(-); !!! не может быть нулевой
   
   QVector<double> qvdWeights ; //!< Weights of each element during training. Are used and reweighted on each iteration
   QVector<QVector<int> > qvqvIntIndexes ; //!< Indexes of each element in training set after sorting
};

/**
CAdaBoostThresholdClassifierCumSum - implementation of adaBoost classifier with cumulative sum usage
*/
class ABC_DLL CAdaBoostThresholdClassifierCumSum : public ABinaryClassifier {  // Derived class
  public: ;
   //! The set of weak classifiers
   QVector<CBinaryWeakThresholdClassifierCumSum> qvWeakClassifier ;

  public: ;
   CAdaBoostThresholdClassifierCumSum () ;
   virtual ~CAdaBoostThresholdClassifierCumSum () ;

   //! Training
   virtual int training (QVector<tDescriptor> const& qvdFirst, QVector<tDescriptor> const& qvdSecond) ;  //!< returns success(0)|fault(!0)

   //! Testing. Classifier MUST be trained before testing
   virtual int testing  (tDescriptor const& desc )const ;  //!< returns rezult of testing {1,-1}
   QVector<int> testing  (QVector<tDescriptor> const& desc)const ;  //!< returns QVector of rezults of testing {1,-1}

   //! Compute voiting weights alpha_m
   virtual double dCalcAlpha_m ( double error_m )const ; //!< returns computed alpha_m

   //! Recompute weights of training data
   virtual void recomputeWeight ( QVector<int> const& pLabels, QVector<int> const& hm, QVector<double> & w_d, double dalpha_m, double errorm )const ; // const!?

   //! Returns indexes of given array after sorting
   static QVector< QVector<int> > qvqviSortByEachDim ( QVector<QVector<double> > const& qvqvdCoords ) ;

   //! Labeling. Separates N elements into iFirstN elements with the value firstValue (1) and N - iFirstN elements with the value secondValue(-1)
   //! Returns labeled vector. N - num of all elements. firstValue - the label of the iFirstN elements. secondValue - the label of the rest.
   static QVector<int> qviSeparate (int iFirstN, int N, int firstValue, int secondValue ) ;

   //! вычислет ошибку классификации слабого классификатора на основе текущих коеффициентов каждого элемента выборки
   virtual double dComputeError ( QVector<int> const& pLabels, QVector<int> const& h, QVector<double> const& qvdWeight )const ; // returns computed error

   //! calculate qvdFirstOut, qvdSecondOut, qvdFirstOutScore, qvdSecondOutScore
   void computeEstimations (QVector<int> const& h_m, QVector<int> const& qvIntLabels, int iNofFirstSet ) ;

   // Function for access to private data
   virtual QVector<int> qviGetFirstLabels () const {return qvdFirstOut; };
   virtual QVector<int> qviGetSecondLabels () const {return qvdSecondOut; };
   virtual QVector<double> qvdGetFirstScores () const {return qvdFirstOutScore; };
   virtual QVector<double> qvdGetSecondScores () const {return qvdSecondOutScore; };
   virtual QVector<double> qvdGetTotalErrors() const {return qvdTotalErrors; };
   void setNumOfIter ( int N ) {iNumOfIter = N;};
   void setDescriptorSize ( int D ) {iDescriptorSize = D;};
   
   // Read/write function. Work with data. Return success(0)|fault(!0)
   virtual int read(QString const& path) ; //!< Reads parameters. Returns success(0)|fault(!0). ISN'T TESTED
   virtual int write(QString const& path) ; //!< Writes parameters. Returns success(0)|fault(!0)

  private: ;
   QVector<int> qvdFirstOut ; //!< The final estimation {1,-1} of each element of negetive set
   QVector<int> qvdSecondOut ; //!< The final estimation {1,-1} of each element of negetive set
   QVector<double> qvdFirstOutScore ; //!< The final scores of each element of positive set
   QVector<double> qvdSecondOutScore ; //!< The final scores of each element of negetive set
   QVector<double> qvdEstimation ; // The estimation of current rate of each element. Isn't used by users
   QVector<double> qvdTotalErrors ;  //!< total error on each iteration
   int iNumOfIter ;  //!< The number of ttraining iterations
   int iDescriptorSize ;  //!< The size of descriptor

   QVector<double> qvdLm ; //!< коэффициент, вес слабого классификатора
};

#endif
