/*!
 *
 * \file adaboost_cumsum_lib.h
 *
 * \brief Implementation of adaptive boosting binary classifier (Adaboost).
 *
 * \authors Evgeny Shtanov
 */

#ifndef __ADABOOSTLIB_CUMSUM_H__
#define __ADABOOSTLIB_CUMSUM_H__
#include "abc_export.h"
#include <QVector>
#include <QString>

// Constant for comparison with zero to predict division by zero.
static const double dEps = 1e-6;

typedef QVector<double> tDescriptor;


/**
 * Abstract binary classifier class.
 */
class ABC_DLL ABinaryClassifier {
  public:;
    virtual ~ABinaryClassifier ();

    //! Any classifier has to be trained before use. Returns success(0) or fault(!0).
    virtual int training ( QVector<tDescriptor> const& qvdFirst, QVector<tDescriptor> const& qvdSecond ) = 0;

    //! Any classifier has testing function. Returns result of testing {1,-1}.
    virtual int testing  ( tDescriptor const& qvdArray ) const = 0;
};


/**
 * The simplest weak binary classifier.
 */
class ABC_DLL CBinaryWeakThresholdClassifierCumSum : public ABinaryClassifier {
  public:;
    CBinaryWeakThresholdClassifierCumSum (void);
    virtual ~CBinaryWeakThresholdClassifierCumSum (void);

    // Initialization.
    //! Set current weights.
    void setWeights( QVector<double> const& qvdW) { qvdWeights = qvdW; };
    //! Set Indices of sorted training set.
    void setSortedIndices( QVector<QVector<int> > const& qvdI) { qvqvIntIndices = qvdI; };
    //! Set current weights and Indices of sorted training set
    void setParam( QVector<double> const& qvdW, QVector<QVector<int> > const& qvdI) { qvdWeights = qvdW; qvqvIntIndices = qvdI; };

    //! Classifier training function.
    virtual int training ( QVector<tDescriptor> const& qvdFirst, QVector<tDescriptor> const& qvdSecond );  //!< returns success(0)|fault(!0)

    //! Testing. Classifier MUST be trained before testing. Returns result of testing {1,-1}.
    virtual int testing ( tDescriptor const& qvdArray ) const;

    //! Functions providing access to private data.
    void setThreshold ( double t ) { Th = t; };
    void setDim ( int d ) { Dim = d; };
    double dGetThreshold ( ) const { return Th; };
    int iGetDim ( ) const { return Dim; };

  private:;
    double Th; //!< Value of coordinate of threshold.
    int Dim;   //!< Dimension of the threshold. Sign determines direction of separation. (Which side is for positive and negative samples).

    QVector<double> qvdWeights; //!< Weights of each element during training. Are used and re-weighed on each iteration.
    QVector<QVector<int> > qvqvIntIndices; //!< Indices of each element in training set after sorting.
};


/**
 * CAdaBoostThresholdClassifierCumSum - implementation of adaBoost classifier with cumulative sum usage.
 */
class ABC_DLL CAdaBoostThresholdClassifierCumSum : public ABinaryClassifier {
  private:;
    // Set of weak classifiers.
    QVector<CBinaryWeakThresholdClassifierCumSum> qvWeakClassifier;

  public:;
    CAdaBoostThresholdClassifierCumSum();
    virtual ~CAdaBoostThresholdClassifierCumSum();

    //! Training. Returns success(0) or fault(!0).
    virtual int training ( QVector<tDescriptor> const& qvdFirst, QVector<tDescriptor> const& qvdSecond );

    //! Testing. Classifier has to be trained before testing.
    //! Returns result of testing {1,-1}.
    virtual int testing(tDescriptor const& desc )const;
    //! Returns QVector of results of testing {1,-1}.
    QVector<int> testing(QVector<tDescriptor> const& desc) const;

    // Read/write function. Work with data.
    //!< Reads parameters. Returns success(0) or fault(!0). ISN'T TESTED
    virtual int read(QString const& path);
    //!< Writes parameters. Returns success(0) or fault(!0).
    virtual int write(QString const& path);

    //! Function for access to private data.
    virtual QVector<int> qviGetFirstLabels () const { return qvdFirstOut; };
    virtual QVector<int> qviGetSecondLabels () const { return qvdSecondOut; };
    virtual QVector<double> qvdGetFirstScores () const { return qvdFirstOutScore; };
    virtual QVector<double> qvdGetSecondScores () const { return qvdSecondOutScore; };
    virtual QVector<double> qvdGetTotalErrors () const { return qvdTotalErrors; };
    void setNumOfIter ( int N ) { iNumOfIter = N; };
    void setDescriptorSize ( int D ) { iDescriptorSize = D; };

  private:;
    //! Compute voting weights alpha_m.
    virtual double dCalcAlpha_m ( double error_m ) const;

    //! Recompute weights of training data.
    virtual void recomputeWeight ( QVector<int> const& pLabels, QVector<int> const& hm, QVector<double> & w_d, double dalpha_m, double errorm ) const;

    //! Returns Indices of given array after sorting
    static QVector< QVector<int> > qvqviSortByEachDim ( QVector<QVector<double> > const& qvqvdCoords );

    //! Labeling. Separates N elements into iFirstN elements with the value firstValue (1) and N - iFirstN elements with the value secondValue(-1)
    //! Returns labeled vector. N - num of all elements. firstValue - the label of the iFirstN elements. secondValue - the label of the rest.
    static QVector<int> qviSeparate (int iFirstN, int N, int firstValue, int secondValue );

    //! Function for error computation.
    virtual double dComputeError ( QVector<int> const& pLabels, QVector<int> const& h, QVector<double> const& qvdWeight ) const;

    //! Calculate qvdFirstOut, qvdSecondOut, qvdFirstOutScore, qvdSecondOutScore.
    void computeEstimations ( QVector<int> const& h_m, QVector<int> const& qvIntLabels, int iNofFirstSet );

  private:;
    QVector<int> qvdFirstOut; //!< The final estimation {1,-1} of each element of positive set.
    QVector<int> qvdSecondOut; //!< The final estimation {1,-1} of each element of negative set.
    QVector<double> qvdFirstOutScore; //!< The final scores of each element of positive set.
    QVector<double> qvdSecondOutScore; //!< The final scores of each element of negative set.
    QVector<double> qvdEstimation; // The estimation of current rate of each element. Isn't used by users.
    QVector<double> qvdTotalErrors;  //!< total error on each iteration.
    int iNumOfIter;
    int iDescriptorSize;
    // Voting weights.
    QVector<double> qvdLm;
};

#endif
