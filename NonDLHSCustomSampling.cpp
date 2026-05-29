#include "NonDLHSCustomSampling.hpp"

namespace Dakota {
void NonDLHSCustomSampling::pre_run()
{
  NonDSampling::pre_run();

  bool increm_lhs_active
    = ( ( sampleType == SUBMETHOD_LHS || 
            sampleType == SUBMETHOD_LOW_DISCREPANCY_SAMPLING ) &&
              !refineSamples.empty() );

  resize_final_statistics_gradients(); // finalStats ASV available at run time

  // BMA TODO: D-optimal incremental LHS (challenging due to set/get ranks)

  // BMA TODO: resolve interaction between VBD and batch sampling
  // (need to generate the VBD replicates for each batch instead of
  // doing VBD on each sequences of samples in order to properly
  // detect duplicates); probably this means this pre_run code
  // migrates to another get_parameter_sets variant that VBD can call

  // Only need to create the pick-freeze samples for the 
  // Saltelli method.
  if (vbdFlag && vbdViaSamplingMethod==VBD_PICK_AND_FREEZE ) {
    get_vbd_parameter_sets(iteratedModel, numSamples);
    if (increm_lhs_active){
      Cout << "\nError: Dakota doesn't currently support incremental sampling with "
      << "the 'variance_based_decomp'\nmethod of type 'pick_and_freeze' active. The "
      << "study will proceed with the original number of\nsamples specified by the 'samples'"
      << "keyword." << std::endl;
      abort_handler(METHOD_ERROR);
    }
    return;
  }

  // DataFitSurrModel sets subIteratorFlag; if true it will manage
  // batch increments 
  // BMA TODO: refactor to handle increments more gracefully
  bool sample_all_batches = !subIteratorFlag;

  // Initial numSamples may be augmented by 1 or more sets of refineSamples
  int seq_len = 1;
  if (sample_all_batches)
    seq_len += refineSamples.length();
  // the user may have fixed the seed; we have to advance it
  if (refineSamples.length() > 0)
    varyPattern = true;

  IntVector samples_vec(seq_len);
  samples_vec[0] = numSamples;
  if (sample_all_batches)
    copy_data_partial(refineSamples, samples_vec, 1);

  // BMA TODO: VBD and other functions aren't accounting for string variables
  // Sampling supports modes beyond just active... do member
  // variable counts suffice?
  size_t cv_start, num_cv, div_start, num_div, dsv_start, num_dsv,
    drv_start, num_drv;
  mode_counts(iteratedModel->current_variables(), cv_start, num_cv, div_start,
	      num_div, dsv_start, num_dsv, drv_start, num_drv);
  size_t num_vars = num_cv + num_div + num_dsv + num_drv;
  int previous_samples = 0, total_samples = samples_vec.normOne();
    
  // Initialize allSamples and all_ranks for total sample size
  if (allSamples.numRows() != num_vars || 
      allSamples.numCols() != total_samples)
    allSamples.shape(num_vars, total_samples);
  IntMatrix all_ranks;
  if (increm_lhs_active)
    all_ranks.shape(num_vars, total_samples);

  for (int batch_ind = 0; batch_ind < seq_len; ++batch_ind) {

    // generate samples of each batch size to reproduce the series
    // of increments, including the point selection
    int new_samples = samples_vec[batch_ind];

    if (increm_lhs_active) {
      // CASE: incremental LHS
      // BMA TODO: allow each batch to be D-optimal w.r.t. previous batch
      if (batch_ind == 0)
	initial_increm_lhs_set(new_samples, allSamples, all_ranks);
      else 
	increm_lhs_parameter_set(previous_samples, new_samples, 
				 allSamples, all_ranks);
    }
    else if (dOptimal) {
      // CASES: random, incremental random, LHS w/ D-optimal
      // populate the correct subset of allSamples, preserving previous
      d_optimal_parameter_set(previous_samples, new_samples, allSamples);
    }
    else {
      // CASES: random, incremental random, LHS
      // sub-matrix of allSamples to populate
      RealMatrix batch_samples(Teuchos::View, allSamples, 
			       num_vars, new_samples,  // num row/col
			       0, previous_samples);   // start row/col
      get_parameter_sets(iteratedModel, new_samples, batch_samples);
    }
    previous_samples += new_samples;
  }
};
}