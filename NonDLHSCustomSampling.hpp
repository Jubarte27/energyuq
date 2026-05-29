/*  _______________________________________________________________________

    Dakota: Explore and predict with confidence.
    Copyright 2014-2025
    National Technology & Engineering Solutions of Sandia, LLC (NTESS).
    This software is distributed under the GNU Lesser General Public License.
    For more information, see the README file in the top Dakota directory.
    _______________________________________________________________________ */

#ifndef NOND_LHS_CUSTOM_SAMPLING_H
#define NOND_LHS_CUSTOM_SAMPLING_H

#include "NonDLHSSampling.hpp"
#include "DataMethod.hpp"


namespace Dakota {

/// Performs LHS and Monte Carlo sampling for uncertainty quantification.

/** The Latin Hypercube Sampling (LHS) package from Sandia
    Albuquerque's Risk and Reliability organization provides
    comprehensive capabilities for Monte Carlo and Latin Hypercube
    sampling within a broad array of user-specified probabilistic
    parameter distributions.  It enforces user-specified rank
    correlations through use of a mixing routine.  The NonDLHSSampling
    class provides a C++ wrapper for the LHS library and is used for
    performing forward propagations of parameter uncertainties into
    response statistics. 

    Batch generation options, including D-Optimal and incremental LHS
    are provided.

    The incremental LHS sampling capability allows one to supplement
    an initial sample of size n to size 2n while maintaining the
    correct stratification of the 2n samples and also maintaining the
    specified correlation structure.  The incremental version of LHS
    will return a sample of size n, which when combined with the
    original sample of size n, allows one to double the size of the
    sample.
*/
class NonDLHSSCustomSampling: public NonDLHSSampling
{
public:

  void pre_run() override;

};

} // namespace Dakota

#endif
