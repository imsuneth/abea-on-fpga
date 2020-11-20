#include "helper.h"

//******************************************************************************************************
/*pre kernel*/
//******************************************************************************************************
__attribute__((reqd_work_group_size(128, 1, 1))) __kernel void
align_kernel_pre_2d(
    __global char *restrict read, __global int32_t *restrict read_len,
    __global ptr_t *restrict read_ptr, __global int32_t *restrict n_events,
    __global ptr_t *restrict event_ptr, __global model_t *restrict models,
    int32_t n_bam_rec, __global model_t *restrict model_kmer_caches,
    __global float *restrict bands1, __global uint8_t *restrict trace1,
    __global EventKmerPair *restrict band_lower_left1) {
  //   printf("Kernel called\n");
  // CUDA
  // int i = blockDim.y * blockIdx.y + threadIdx.y;
  // int tid=blockIdx.x*blockDim.x+threadIdx.x;

  size_t i = get_global_id(1);
  size_t tid = get_global_id(0);

  if (i < n_bam_rec) {
    __global char *sequence = &read[read_ptr[i]];
    int32_t sequence_len = read_len[i];
    // int32_t n_event = n_events[i];
    __global model_t *model_kmer_cache = &model_kmer_caches[read_ptr[i]];
    __global float *bands =
        &bands1[(read_ptr[i] + event_ptr[i]) * ALN_BANDWIDTH];
    __global uint8_t *trace =
        &trace1[(read_ptr[i] + event_ptr[i]) * ALN_BANDWIDTH];
    __global EventKmerPair *band_lower_left =
        &band_lower_left1[read_ptr[i] + event_ptr[i]];

    // int32_t n_events = n_event;
    int32_t n_kmers = sequence_len - KMER_SIZE + 1;
    // fprintf(stderr,"n_kmers : %d\n",n_kmers);

    // transition penalties
    // float events_per_kmer = (float)n_events / n_kmers;
    // float p_stay = 1 - (1 / (events_per_kmer + 1));

    // setting a tiny skip penalty helps keep the true alignment within the
    // adaptive band this was empirically determined double epsilon = 1e-10;
    // double lp_skip = log(epsilon);
    // double lp_stay = log(p_stay);
    // double lp_step = log(1.0 - exp(lp_skip) - exp(lp_stay));
#ifndef ALIGN_KERNEL_FLOAT
    double lp_trim = log(0.01);
#else
    float lp_trim = logf(0.01f);
#endif

    // dp matrix
    // int32_t n_rows = n_events + 1;
    // int32_t n_cols = n_kmers + 1;
    // int32_t n_bands = n_rows + n_cols;

    // Initialize
    // Precompute k-mer ranks to avoid doing this in the inner loop

    // #ifdef  PRE_3D
    //     if(band_i<n_kmers && band_j==0){
    // #else
    //     if(band_i<n_kmers){
    // #endif

    if (tid == 0) { // todo : can be optimised
      for (int32_t i = 0; i < n_kmers; ++i) {
        // kmer_ranks[i] = get_kmer_rank(substring, KMER_SIZE);
        __global char *substring = &sequence[i];

        uint32_t kmer_ranks = get_kmer_rank(substring, KMER_SIZE);
        model_kmer_cache[i] = models[kmer_ranks];
      }
    }

    if (tid < bandwidth) {
      for (int32_t i = 0; i < 3; i++) {
        BAND_ARRAY(i, tid) = -INFINITY;
        // TRACE_ARRAY(i,tid) = 0;
      }
    }

    if (tid == 0) {
      // initialize range of first two bands
      band_lower_left[0].event_idx = half_bandwidth - 1;
      band_lower_left[0].kmer_idx = -1 - half_bandwidth;
      // band_lower_left[1] = move_down(band_lower_left[0]);
      band_lower_left[1].event_idx = band_lower_left[0].event_idx + 1;
      band_lower_left[1].kmer_idx = band_lower_left[0].kmer_idx;

      int start_cell_offset = band_kmer_to_offset(0, -1);
      // assert(is_offset_valid(start_cell_offset));
      // assert(band_event_to_offset(0, -1) == start_cell_offset);
      BAND_ARRAY(0, start_cell_offset) = 0.0f;

      // band 1: first event is trimmed
      int first_trim_offset = band_event_to_offset(1, 0);
      // assert(kmer_at_offset(1, first_trim_offset) == -1);
      // assert(is_offset_valid(first_trim_offset));
      BAND_ARRAY(1, first_trim_offset) = lp_trim;
      TRACE_ARRAY(1, first_trim_offset) = FROM_U;

      // int fills = 0;
#ifdef DEBUG_ADAPTIVE
          fprintf(stderr, "[trim] bi: %d o: %d e: %d k: %d s: %.2lf\n", 1,
                  first_trim_offset, 0, -1, BAND_ARRAY(1,first_trim_offset);
#endif
    }
  }
}
