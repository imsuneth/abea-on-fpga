
#include "f5c.h"
// #include <assert.h>
// #include <assert.h>
// #include "f5cmisc.cuh"

//#define DEBUG_ESTIMATED_SCALING 1
//#define DEBUG_RECALIB_SCALING 1
//#define DEBUG_ADAPTIVE 1
// #define CACHED_LOG 1

// From f5cmisc.cuh*****************************
// #define ALIGN_KERNEL_FLOAT 1 //(for 2d kernel only)
// #define WARP_HACK 1 // whether the kernels are  performed in 1D with a warp
// hack (effective only  if specific TWODIM_ALIGN is not defined)
// #define REVERSAL_ON_CPU \
//   1 // reversal of the backtracked array is performed on the CPU instead of
//   the
//     // GPU

//*********************************************
#define bandwidth ALN_BANDWIDTH
#define half_bandwidth ALN_BANDWIDTH / 2

#define log_inv_sqrt_2pi -0.918938f
#define max_gap_threshold 50
#define min_average_log_emission -5.0
#define epsilon 1e-10

#define FROM_D 0
#define FROM_U 1
#define FROM_L 2

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define is_offset_valid(offset) (offset) >= 0 && (offset) < bandwidth
// #ifdef ALIGN_2D_ARRAY
// #define BAND_ARRAY(r, c) (bands[(r)][(c)])
// #define TRACE_ARRAY(r, c) (trace[(r)][(c)])
// #else
#define BAND_ARRAY(r, c) (bands[((r) * (ALN_BANDWIDTH) + (c))])
#define TRACE_ARRAY(r, c) (trace[((r) * (ALN_BANDWIDTH) + (c))])
// #endif

// __attribute__((max_work_group_size(1024)))
//
// __attribute__((reqd_work_group_size(1, 1, 1)))
__attribute__((num_compute_units(1))) __attribute__((task)) __kernel void
align_kernel_single(
    // __global AlignedPair *restrict event_align_pairs,
    // __global int32_t *restrict n_event_align_pairs,
    // __global char *restrict read,
    __global int32_t *restrict read_len, __global ptr_t *restrict read_ptr,
    __global event1_t *restrict event_table,
    __global int32_t *restrict n_events1, __global ptr_t *restrict event_ptr,
    __global scalings_t *restrict scalings, __global model_t *restrict models,
    int32_t n_bam_rec, __global uint32_t *restrict kmer_ranks1,
    __global float *restrict band, __global uint8_t *restrict traces,
    __global EventKmerPair *restrict band_lower_lefts) {

// size_t ii = get_global_id(0);
// printf("START!!!!!!!!!!!!!!!\n");
#pragma ii 1
#pragma ivdep
  for (int32_t ii = 0; ii < n_bam_rec; ii++) {

    int32_t sequence_len = read_len[ii];
    __global event1_t *events = event_table + event_ptr[ii];
    int32_t n_events = n_events1[ii];
    scalings_t scaling = scalings[ii];
    __global float *bands =
        band + (read_ptr[ii] + event_ptr[ii]) * ALN_BANDWIDTH;
    __global uint8_t *trace =
        traces + (read_ptr[ii] + event_ptr[ii]) * ALN_BANDWIDTH;
    __global EventKmerPair *band_lower_left =
        band_lower_lefts + read_ptr[ii] + event_ptr[ii];
    __global uint32_t *kmer_ranks = kmer_ranks1 + read_ptr[ii];

    int32_t n_kmers = sequence_len - KMER_SIZE + 1;

    // transition penalties
    double events_per_kmer = (double)n_events / n_kmers;
    double p_stay = 1 - (1 / (events_per_kmer + 1));

    // setting a tiny skip penalty helps keep the true alignment within the
    // adaptive band this was empirically determined

    float lp_skip = log(epsilon);
    float lp_stay = log(p_stay);
    float lp_step = log(1.0 - exp(lp_skip) - exp(lp_stay));
    float lp_trim = log(0.01);
    // printf("lp_step %lf \n", lp_step);

    // dp matrix
    int32_t n_rows = n_events + 1;
    int32_t n_cols = n_kmers + 1;
    int32_t n_bands = n_rows + n_cols;

    // fill in remaining bands
    // #pragma unroll
    bool odd_band_idx = true;
    for (int32_t band_idx = 2; band_idx < n_bands; ++band_idx) {
      odd_band_idx = !odd_band_idx;
      // if (band_idx < n_bands) {

      // Determine placement of this band according to Suzuki's adaptive
      // algorithm When both ll and ur are out-of-band (ob) we alternate
      // movements otherwise we decide based on scores
      float ll = BAND_ARRAY(band_idx - 1, 0);
      float ur = BAND_ARRAY(band_idx - 1, bandwidth - 1);
      bool ll_ob = ll == -INFINITY;
      bool ur_ob = ur == -INFINITY;

      bool right = false;
      if (ll_ob && ur_ob) {
        // right = band_idx % 2 == 1;
        right = odd_band_idx;
      } else {
        right = ll < ur; // Suzuki's rule
      }
      EventKmerPair bbl = band_lower_left[band_idx - 1];
      if (right) {

        bbl.kmer_idx++;

      } else {

        bbl.event_idx++;
      }
      band_lower_left[band_idx] = bbl;

      int trim_offset = (-1) - bbl.kmer_idx;
      // printf("%d ", trim_offset);
      if (is_offset_valid(trim_offset)) {
        int32_t event_idx = bbl.event_idx - (trim_offset);
        // printf("%ld ", event_idx);
        if (event_idx >= 0 && event_idx < n_events) {
          BAND_ARRAY(band_idx, trim_offset) = lp_trim * (event_idx + 1);
          TRACE_ARRAY(band_idx, trim_offset) = FROM_U;
        } else {
          BAND_ARRAY(band_idx, trim_offset) = -INFINITY;
        }
      }

      // Get the offsets for the first and last event and kmer
      // We restrict the inner loop to only these values
      // printf("%d ", band_lower_left[band_idx].kmer_idx);
      // printf("%d ", band_idx);

      // int kmer_min_offset = band_kmer_to_offset(band_idx, 0);
      int kmer_min_offset = 0 - bbl.kmer_idx;
      int kmer_max_offset = n_kmers - bbl.kmer_idx;
      int event_min_offset =
          bbl.event_idx - (n_events - 1);
      int event_max_offset = bbl.event_idx - (-1);

      int min_offset = MAX(kmer_min_offset, event_min_offset);
      min_offset = MAX(min_offset, 0);

      int max_offset = MIN(kmer_max_offset, event_max_offset);
      max_offset = MIN(max_offset, bandwidth);

      // #pragma speculated_iterations 3
      // #pragma unroll
      for (int offset = 0; offset < ALN_BANDWIDTH; ++offset) {

        if (offset >= min_offset && offset < max_offset) {

          int event_idx = bbl.event_idx - offset;
          int kmer_idx = bbl.kmer_idx + offset;

          int32_t kmer_rank = kmer_ranks[kmer_idx];
          // printf("%ld ", kmer_rank);

          //===================================================================
          // event level mean, scaled with the drift value
          // strand = 0;
          // assert(kmer_rank < 4096);
          // float level = read.get_drift_scaled_level(event_idx, strand);

          // float time =
          //    (events.event[event_idx].start - events.event[0].start) /
          //    sample_rate;
          // float unscaledLevel = events.event[event_idx].mean;
          float unscaledLevel = events[event_idx].mean;
          float scaledLevel = unscaledLevel;
          // float scaledLevel = unscaledLevel - time * scaling.shift;

          // fprintf(stderr, "level %f\n",scaledLevel);
          // GaussianParameters gp =
          // read.get_scaled_gaussian_from_pore_model_state(pore_model, strand,
          // kmer_rank);
          float gp_mean =
              scaling.scale * models[kmer_rank].level_mean + scaling.shift;
          float gp_stdv = models[kmer_rank].level_stdv * 1; // scaling.var = 1;
// float gp_stdv = 0;
// float gp_log_stdv = models[kmer_rank].level_log_stdv + scaling.log_var;
// if(models[kmer_rank].level_stdv <0.01 ){
// 	fprintf(stderr,"very small std dev %f\n",models[kmer_rank].level_stdv);
// }
#ifdef CACHED_LOG
          float gp_log_stdv = models[kmer_rank].level_log_stdv;
#else
          float gp_log_stdv =
              log(models[kmer_rank].level_stdv); // scaling.log_var = log(1)=0;
#endif

          /*INCOMPLETE*/
          float a = (scaledLevel - gp_mean) / gp_stdv;
          // return 1;

          float lp_emission = log_inv_sqrt_2pi - gp_log_stdv + (-0.5f * a * a);

          //==================================================================

          int offset_up =
              band_lower_left[band_idx - 1].event_idx - (event_idx - 1);
          int offset_left =
              (kmer_idx - 1) - band_lower_left[band_idx - 1].kmer_idx;
          int offset_diag =
              (kmer_idx - 1) - band_lower_left[band_idx - 2].kmer_idx;

          float up = is_offset_valid(offset_up)
                         ? BAND_ARRAY(band_idx - 1, offset_up)
                         : -INFINITY;
          float left = is_offset_valid(offset_left)
                           ? BAND_ARRAY(band_idx - 1, offset_left)
                           : -INFINITY;
          float diag = is_offset_valid(offset_diag)
                           ? BAND_ARRAY(band_idx - 2, offset_diag)
                           : -INFINITY;

          // fprintf(stderr, "lp emiision : %f , event idx %d, kmer rank
          // %d\n", lp_emission,event_idx,kmer_rank);
          float score_d = diag + lp_step + lp_emission;
          float score_u = up + lp_stay + lp_emission;
          float score_l = left + lp_skip;

          float max_score = score_d;
          uint8_t from = FROM_D;

          max_score = score_u > max_score ? score_u : max_score;
          from = max_score == score_u ? FROM_U : from;
          max_score = score_l > max_score ? score_l : max_score;
          from = max_score == score_l ? FROM_L : from;

          BAND_ARRAY(band_idx, offset) = max_score;
          // printf("%f ", max_score);
          TRACE_ARRAY(band_idx, offset) = from;
          // fills += 1;
        }
      }
      // } // if (band_idx < n_bands)
      // else {
      //   break;
      // }
    } // for (int32_t band_idx = 2; band_idx < 100000; ++band_idx)
  }
}