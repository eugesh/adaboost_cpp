#include "adaboost_cumsum_lib.h"
#include "dop_func.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <QFile>
#include <QTextStream>


/*
 * Internal structure. It is used for sorting training/testing set by coordinates.
 */
struct TrainingSetNumerator {  // use for sorting
    int number; // Number of element in training set.
    double point;  // Value of current coordinate in training set.
    TrainingSetNumerator() { ; };
    TrainingSetNumerator(int n, double p) : number(n), point(p) { ; }
};

/*
 * Sort by coordinate. Internal function.
 */
bool lessThen (const TrainingSetNumerator &a1, const TrainingSetNumerator &a2);

/*
 * Auxilliary functions.
 */

bool
lessThen (const TrainingSetNumerator &a1, const TrainingSetNumerator &a2) {
   return a1.point <= a2.point;
}

double
maxQV( QVector<double> array ) {
   double xMax = array[0];

   for(int i=0; i < array.size(); ++i) {
      if(xMax < array[i]) {
         xMax = array[i];
      }
   }

   return xMax;
}

double
minQV( QVector<double> array ) {
   double xMin = array[0];

   for(int i=0; i < array.size(); ++i) {
      if(xMin > array[i]) {
         xMin = array[i];
      }
   }
   return xMin;
}

double
maxQV( QVector<double> array , int &position ) {
   double xMax = array[0];
   position = 0;

   for(int i=0; i < array.size(); ++i) {
      if(xMax < array[i]) {
         xMax = array[i];
         position = i;
      }
   }
   return xMax;
}

double
minQV( QVector<double> array, int &position ) {
   double xMin = array[0];
   position = 0;

   for(int i=0; i < array.size(); ++i) {
      if(xMin > array[i]) {
         xMin = array[i];
         position = i;
      }
   }
   return xMin;
}

// Abstract class destructor (Interfaces)
ABinaryClassifier::~ABinaryClassifier() {
   // cout<<"ABinaryClassifier::Dstr" << endl;
}

// Derived classes constructors and destructors  (Realisations)
CBinaryWeakThresholdClassifierCumSum::CBinaryWeakThresholdClassifierCumSum (void)
{
   // cout<<"CBinaryWeakThresholdClassifierCumSum::Cstr" << endl;
}

CBinaryWeakThresholdClassifierCumSum::~CBinaryWeakThresholdClassifierCumSum (void) {
   // cout<<"CBinaryWeakThresholdClassifierCumSum::Dstr" << endl;
}

CAdaBoostThresholdClassifierCumSum::CAdaBoostThresholdClassifierCumSum()
  : iNumOfIter(1), iDescriptorSize(1) {
   // cout<<"CAdaBoostThresholdClassifierCumSum::Cstr"<<endl;
}

CAdaBoostThresholdClassifierCumSum::~CAdaBoostThresholdClassifierCumSum() {
   // cout<<"CAdaBoostThresholdClassifierCumSum::Dstr"<<endl;
}

int
CBinaryWeakThresholdClassifierCumSum::testing(QVector<double> const& qvdArray) const {
   int rez;

   if (sign(Dim) > 0) {
     if (qvdArray[std::abs(Dim) - 1] < Th || (qvdArray[std::abs(Dim) - 1] - Th) < dEps)
        rez =  1;
     else
        rez = -1;
   } else {
     if (qvdArray[std::abs(Dim) - 1] > Th)
        rez =  1;
     else
        rez = -1;
   }

   return rez;
}

void
CAdaBoostThresholdClassifierCumSum::recomputeWeight(QVector<int> const& pLabels,
                                                    QVector<int> const& h_m,
                                                    QVector<double> & w_d,
                                                    double dalpha_m,
                                                    double errorm) const {
   int iN;
   iN = pLabels.size();

   for (int i = 0; i < iN; ++i) {
      w_d[i] *= exp(-std::max(dalpha_m, dEps) * h_m[i] * pLabels[i]); // recalculation of D weights
   }

   // sum of w_d
   double sum_w = 0.0;
   for (int i=0; i < iN; ++i)
      sum_w += w_d[i];

   for (int i=0; i < iN; ++i) {
      w_d[i] /= std::max(sum_w, dEps); // sum of weights should be equal to 1 , reweighting
   }
}

double
CAdaBoostThresholdClassifierCumSum::dComputeError ( QVector<int> const& pLabels, QVector<int> const& h_m, QVector<double> const& qvdWeight )const { // returns computed error
   double error_m = 0;

   std::cout << " h_m[i] << pLabels[i]: " << std::endl;
   for(int i=0; i < pLabels.size(); ++i) {
      std::cout << h_m[i] << " " << pLabels[i] << std::endl;
      // Compute error of current weak clissifier h_func_m.
      error_m += std::abs(qvdWeight[i] * (h_m[i] != pLabels[i]));
   }

   return error_m;
}

double
CAdaBoostThresholdClassifierCumSum::dCalcAlpha_m ( double error_m ) const {
   return ( 0.5 ) * log ( ( 1 - error_m ) / std::max( error_m, dEps ) );
}

/**
 * Training function.
 *
 * \param qvdFirst vector of features of positive class;
 * \param qvdSecond vector of features of negative class. Size of qvdFirst and qvdSecond has to be the same.
 */
int
CAdaBoostThresholdClassifierCumSum::training( QVector<tDescriptor> const& qvdFirst, QVector<tDescriptor> const& qvdSecond ) {
    if( qvdFirst.isEmpty() || qvdSecond.isEmpty() ) {
        std::cout << "Error: empty training data.";
        return -1;
    }
    if( qvdFirst[0].size() != qvdSecond[0].size() ) {
        std::cout << "Error: Incorrect training data.";
        return -1;
    }

    iDescriptorSize = qvdFirst[0].size();

    // Training set.
    QVector<QVector<double> > qvqvdTrainSet = qvdFirst + qvdSecond;
    int iNumOfElements = qvqvdTrainSet.size();

    // Indices of each element in the trainig set after sorting.
    QVector<QVector<int> > qvqviIndices;

    // Resize estimation array. Estimations are accumulated on each training iteration.
    qvdEstimation.resize(iNumOfElements);

    qvqviIndices.resize(iNumOfElements);

    // Preparations.
    // Sorting.
    qvqviIndices = qvqviSortByEachDim( qvqvdTrainSet );
    // Labeling.
    QVector<int> qvIntLabels = qviSeparate( qvdFirst.size(), iNumOfElements, 1, -1 );

    // Cumulative sums.
    QVector<double> qvdCumSum;
    // Weights of each element.
    QVector<double> qvdWeights;
    // Weak classifier.
    QVector<int> qviH_m;

    qvdCumSum.resize(iNumOfElements);
    qviH_m.resize(iNumOfElements);

    for ( int i = 0; i < iNumOfElements; ++i ) {
        qvdWeights << 1.0 / iNumOfElements;
    }
    qvWeakClassifier.resize(iNumOfIter);

    // Adaboost iteration cycle.
    for ( int m = 0; m < iNumOfIter; ++m ) {
        std::cout << "_____________________________________" << std::endl;
        std::cout << "No error on " << m << " iteration." << std::endl;
        std::cout << "_____________________________________" << std::endl;
        {
            // Weak classifier training.
            qvWeakClassifier[m].setWeights(qvdWeights);
            qvWeakClassifier[m].setSortedIndices(qvqviIndices);

            qvWeakClassifier[m].training(qvdFirst, qvdSecond);
            std::cout << "Weak classifier is trained successfully!" << std::endl;
        }

        for(int j = 0; j < iNumOfElements; ++j)
            qviH_m[j] = qvWeakClassifier[m].testing(qvqvdTrainSet[j]);

        double dError_m = dComputeError(qvIntLabels, qviH_m, qvdWeights);
        std::cout << "dError_m = " << dError_m << std::endl;

        if (std::abs(dError_m) > 0.5) {
            std::cout << "Warning: abs(dError_m[m]) > 0.5!!! Stop training. " << std::endl;  // if current error more than 50% -> stop training
            break;
        }

        qvdLm << dCalcAlpha_m(dError_m);
        std::cout << "qvdLm = " << qvdLm.last() << std::endl;

        recomputeWeight(qvIntLabels, qviH_m, qvdWeights, qvdLm.last(), dError_m);

        computeEstimations(qviH_m, qvIntLabels, qvdFirst.size());
    }

    return 0;
}

/*
 * Computes scores and labels.
 */
void
CAdaBoostThresholdClassifierCumSum::computeEstimations ( QVector<int> const& h_m, QVector<int> const& qvIntLabels, int iNofFirstSet ) {
    int iN;
    iN = h_m.size();
    qvdFirstOut.resize(iNofFirstSet);
    qvdFirstOutScore.resize(iNofFirstSet);
    qvdSecondOut.resize(iN - iNofFirstSet);
    qvdSecondOutScore.resize(iN - iNofFirstSet);

    qvdTotalErrors << 0;  // push new element

    //________Estimation____________
    for (int i = 0; i < iN; ++i) {
        // Calculate estimation of current rate of each element.
        qvdEstimation[i] += qvdLm.last() * h_m[i]; // * sign(pOutStumps[m].Dim);
        if( i < iNofFirstSet  ) {
            // Define class of current element.
            qvdFirstOut[i] = sign(qvdEstimation[i]);
            qvdFirstOutScore[i] = qvdEstimation[i];
        }

        else {
            qvdSecondOut[ i - iNofFirstSet ] = sign(qvdEstimation[i]);
            qvdSecondOutScore[ i - iNofFirstSet ] = qvdEstimation[i];
        }

        if ( sign(qvdEstimation[i]) != qvIntLabels[i] ) {
            qvdTotalErrors.last() += (double) 1 / (double) iN;
        }
    }
}

int
CAdaBoostThresholdClassifierCumSum::testing( tDescriptor const& qvdArray ) const {
    double estimateclasssum_te = 0;

    // Number of iteration cycle.
    for (int m = 0; m < qvWeakClassifier.size(); ++m) {
        // Weak classifier testing.
        int h = qvWeakClassifier[m].testing(qvdArray);

        // Estimation for all trained weak classifiers
        estimateclasssum_te = estimateclasssum_te + qvdLm[m] * h; // * sign( qvWeakClassifier[m].iGetDim() );
    }

    return sign(estimateclasssum_te);
}

QVector<int>
CAdaBoostThresholdClassifierCumSum::testing( QVector<tDescriptor> const& desc ) const {
    QVector<int> qviLabels;

    for(int i = 0; i < desc.size(); ++i)
        qviLabels << testing(desc[i]);

    return qviLabels;
}

/**
 * Sort features in each dimension.
 *
 * \param qvqvdCoords set of features vectors;
 *
 * \return indices of sorted array in input array.
 */
QVector<QVector<int> >
CAdaBoostThresholdClassifierCumSum::qvqviSortByEachDim( QVector<tDescriptor> const& qvqvdCoords ) {
    if(qvqvdCoords.isEmpty()) {
        std::cout << "CAdaBoostThresholdClassifierCumSum::qvqviSortByEachDim : empty input data" << std::endl;
        return QVector<QVector<int> >();
    }

    QVector<QVector<int> > qvqviIndices;

    int iN;
    iN = qvqvdCoords.size();
    int iDescrSize = qvqvdCoords.first().size();

    // Vector of each element in one dimension.
    QVector<TrainingSetNumerator> list;
    qvqviIndices.resize(iN);

    // Cycle in each dimension.
    for ( int d = 0; d < iDescrSize; ++d ) {
        list.clear();
        // Cycle for all elements.
        for ( int i = 0; i < iN; ++i ) {
            TrainingSetNumerator buf( i, qvqvdCoords[i][d] );
            list.push_back(buf);  // positive
        }
        // Sort each element of training data by each dimension.
        qSort(list.begin(), list.end(), lessThen);

        for (int i = 0; i < iN; ++i) {
            // Indices in original array.
            qvqviIndices[i] << list[i].number;
        }
    }

    return  qvqviIndices;
}

/**
 * Separates N elements into iFirstN and N - iFirstN sets of elements.
 *
 * \param iFirstN - number of elements in the first set;
 * \param size - number of elements;
 * \param firstValue;
 * \param secondValue - values of labels (e.g. 1/-1 );
 *
 * \return labeled array.
 */
QVector<int>
CAdaBoostThresholdClassifierCumSum::qviSeparate( int iFirstN, int size, int firstValue, int secondValue ) {
    QVector<int> qvIntLabels;

    // Separation of training set (labeling).
    for (int i = 0; i < size; ++i ) {
        if ( i < iFirstN )
            qvIntLabels << firstValue;
        else
            qvIntLabels << secondValue;
    }

    return qvIntLabels;
}

int
CBinaryWeakThresholdClassifierCumSum::training( QVector<QVector<double> > const & qvdFirst, QVector<QVector<double> > const & qvdSecond ) {
    int iNumOfElements = qvdWeights.size();

    // Cumulative sums.
    QVector<double> qvdCumSum;
    qvdCumSum.resize(iNumOfElements - 1);
    double dMinError = 0;

    // Variables for the best decision after m iterations.
    int iDimension_m = 0;
    int iIndexOfMaxElement_m;
    int iIndexOfMinElement_m;
    double dMaxElement_m;
    double dMinElement_m;

    QVector<int> qviLabels;

    // Training set.
    QVector<QVector<double> > qvqvdTrainSet;
    qvqvdTrainSet = qvdFirst + qvdSecond;

    int iDescriptorSize = qvdFirst[0].size();

    for(int i = 0; i < qvqvdTrainSet.size(); ++i )
        if( i < qvdFirst.size() )
            qviLabels <<  1;   // labeling
        else
            qviLabels << -1;

    {
        // Looking for extremum.
        double xmax, xmin;
        int imax, imin;
        //  Cycle in each dimension.
        for ( int d = 0; d < iDescriptorSize; ++d ) {
            // Cycle for each element.
            // Cumulative sums initialization.
            for ( int i = 0; i < iNumOfElements - 1; ++i ) {
                qvdCumSum[i] = 0;
            }
            // The first element of cumsum.
            qvdCumSum[0] = qvdWeights[ qvqvIntIndices[0][d] ] * qviLabels[ qvqvIntIndices[0][d] ];

            for (int i = 1; i < iNumOfElements - 1; ++i) {
                // Cumulative sum calculation.
                qvdCumSum[i] = qvdCumSum[i - 1] + ( qvdWeights[qvqvIntIndices[i][d]] *
                                             qviLabels[qvqvIntIndices[i][d]] );
            }

            // Find max/min element of cum_sum and index of it => extremum => the best threshold.
            xmax = maxQV(qvdCumSum, imax);
            xmin = minQV(qvdCumSum, imin);

            if( dMinError < std::max( std::abs(xmax), std::abs(xmin) ) ) {
                dMinError = std::max( std::abs(xmax), std::abs(xmin) );
                // Fixate the best dimension and coordinate of the best threshold.
                iDimension_m = d;
                // Index of the best threshold.
                iIndexOfMaxElement_m = imax;
                iIndexOfMinElement_m = imin;
                // Value of the best threshold.
                dMaxElement_m = xmax;
                dMinElement_m = xmin;
            }
        }
    }

    std::cout << "No error after looking for extremum." << std::endl;
    {
        // Threshold selection.
        Dim = iDimension_m + 1;

        std::cout << "Dim = " << Dim << std::endl;
        std::cout << "iIndexOfMaxElement_m = " << iIndexOfMaxElement_m << std::endl;
        std::cout << "iIndexOfMinElement_m = " << iIndexOfMinElement_m << std::endl;
        std::cout << "dMaxElement_m = " << dMaxElement_m << std::endl;
        std::cout << "dMinElement_m = " << dMinElement_m << std::endl;
        std::cout << "qvqvIntIndices[iIndexOfMaxElement_m][iDimension_m] = " << qvqvIntIndices[iIndexOfMaxElement_m][iDimension_m] << std::endl;
        std::cout << "qvqvIntIndices[iIndexOfMinElement_m][iDimension_m] = " << qvqvIntIndices[iIndexOfMinElement_m][iDimension_m] << std::endl;
        // Determine the sign of inequality.
        if (std::abs(dMaxElement_m) > std::abs(dMinElement_m)) {
            Th = qvqvdTrainSet[ qvqvIntIndices[iIndexOfMaxElement_m][iDimension_m] ] [iDimension_m];

            Dim *= 1;  // Forward.
        }
        else {
            Th = qvqvdTrainSet[ qvqvIntIndices[iIndexOfMinElement_m][iDimension_m] ] [iDimension_m];

            Dim *= -1;  // Backward.
        }
    }
    std::cout << "Th = " << Th << std::endl;
    std::cout << "Dim = " << Dim << std::endl;
    std::cout << "No error after threshold selection" << std::endl;

    return 0;
}

/**
 * Reads parameters. Returns success(0) || fault (!0).
 */
int
CAdaBoostThresholdClassifierCumSum::read(QString const& path) {
	QFile file(path);
	if ( !file.open(QIODevice::ReadOnly)) {
		fprintf(stdout, "CBinaryClassifier::Read : Can't read <%s>\n", qPrintable(path));
		return -1;
	};

	double dtemp; // temporal variable

	QVector<double> alpha_vector;
	QVector<double> th_vector;
	QVector<int> dim_vector;

	QTextStream ts(&file);
	ts >> iNumOfIter;
	//ts >> dtemp;
	//ts >> dtemp;
	ts >> iDescriptorSize;

	qvWeakClassifier.clear();
	qvWeakClassifier.resize(iNumOfIter);
	qvdLm.clear();
	qvdLm.resize(iNumOfIter);

	for(int i = 0; i < iNumOfIter; i++)
	ts >> qvdLm[i];

	for(int i = 0; i < iNumOfIter; i++) {
		ts >> dtemp;
		qvWeakClassifier[i].setThreshold(dtemp);
	}

	int itemp;

	for(int i = 0; i < iNumOfIter; i++) {
		ts >> itemp;
		qvWeakClassifier[i].setDim(itemp);
	}

	for(int i = 0; i < iNumOfIter; i++) {
		fprintf(stdout, "%d : %lf %f %d\n", i, qvdLm[i], qvWeakClassifier[i].dGetThreshold(), qvWeakClassifier[i].iGetDim());
	}

	return 0;
}

/**
 * Writes parameters. Returns success(0) || fault (!0).
 */
int
CAdaBoostThresholdClassifierCumSum::write(QString const& path) {
    FILE *fp;
    if((fp = fopen(path.toLocal8Bit().data(), "w")) == NULL) {
        printf("File %s hasn't been opened", path.toLocal8Bit().data());
        return -1;
    }

    fprintf(fp, "%d\n", iNumOfIter);
    fprintf(fp, "%d\n", iDescriptorSize);

    for(int i=0; i < qvdLm.size(); ++i)
        fprintf(fp, "%f ", qvdLm[i]);
    fprintf(fp, "\n");

    for(int i=0; i < qvWeakClassifier.size(); ++i)
        fprintf(fp, "%f ", qvWeakClassifier[i].dGetThreshold( ));
    fprintf(fp, "\n");

    for(int i=0; i < qvWeakClassifier.size(); ++i)
        fprintf(fp, "%d ", qvWeakClassifier[i].iGetDim( ));
    fprintf(fp, "\n");

    fclose(fp);

    return 0;
}
