
#include "f5c.h"

// #include <assert.h>
// #include "f5cmisc.cuh"

//#define DEBUG_ESTIMATED_SCALING 1
//#define DEBUG_RECALIB_SCALING 1
//#define DEBUG_ADAPTIVE 1

// From f5cmisc.cuh*****************************
// #define ALIGN_KERNEL_FLOAT 1 //(for 2d kernel only)
// #define WARP_HACK 1 // whether the kernels are  performed in 1D with a warp
// hack (effective only  if specific TWODIM_ALIGN is not defined)
// #define REVERSAL_ON_CPU \
//   1 // reversal of the backtracked array is performed on the CPU instead of
//   the
//     // GPU

//*********************************************

// todo : can make more efficient using bit encoding
inline uint32_t get_rank(char base) {
  if (base == 'A') { // todo: do we neeed simple alpha?
    return 0;
  } else if (base == 'C') {
    return 1;
  } else if (base == 'G') {
    return 2;
  } else if (base == 'T') {
    return 3;
  } else {
    // WARNING("A None ACGT base found : %c", base);
    return 0;
  }
}

// return the lexicographic rank of the kmer amongst all strings of
// length k for this alphabet
inline uint32_t get_kmer_rank(volatile __global char *str, uint32_t k) {
  // uint32_t p = 1;
  uint32_t r = 0;

// from last base to first
#pragma unroll
  for (uint32_t i = 0; i < k; ++i) {
    // r += rank(str[k - i - 1]) * p;
    // p *= size();
    r += get_rank(str[k - i - 1]) << (i << 1);
  }
  return r;
}

// copy a kmer from a reference
inline void kmer_cpy(char *dest, char *src, uint32_t k) {
  uint32_t i = 0;
  for (i = 0; i < k; i++) {
    dest[i] = src[i];
  }
  dest[i] = '\0';
}

#define log_inv_sqrt_2pi -0.918938f // Natural logarithm

inline float log_normal_pdf(float x, float gp_mean, float gp_stdv,
                            float gp_log_stdv) {
  /*INCOMPLETE*/
  // float log_inv_sqrt_2pi = -0.918938f; // Natural logarithm
  float a = (x - gp_mean) / gp_stdv;
  return log_inv_sqrt_2pi - gp_log_stdv + (-0.5f * a * a);
  // return 1;
}

inline float log_probability_match_r9(scalings_t scaling,
                                      __global model_t *models,
                                      __global event1_t *events, int event_idx,
                                      uint32_t kmer_rank) {
  // event level mean, scaled with the drift value
  // strand = 0;
#ifdef DEBUG_ADAPTIVE
  // assert(kmer_rank < 4096);
#endif
  // float level = read.get_drift_scaled_level(event_idx, strand);

  // float time =
  //    (events.event[event_idx].start - events.event[0].start) / sample_rate;
  float unscaledLevel = events[event_idx].mean;
  float scaledLevel = unscaledLevel;
  // float scaledLevel = unscaledLevel - time * scaling.shift;

  // fprintf(stderr, "level %f\n",scaledLevel);
  // GaussianParameters gp =
  // read.get_scaled_gaussian_from_pore_model_state(pore_model, strand,
  // kmer_rank);
  float gp_mean = scaling.scale * models[kmer_rank].level_mean + scaling.shift;
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

  float lp = log_normal_pdf(scaledLevel, gp_mean, gp_stdv, gp_log_stdv);
  return lp;
}

#define event_kmer_to_band(ei, ki) (ei + 1) + (ki + 1)
#define band_event_to_offset(bi, ei) band_lower_left[bi].event_idx - (ei)
#define band_kmer_to_offset(bi, ki) (ki) - band_lower_left[bi].kmer_idx
#define is_offset_valid(offset) (offset) >= 0 && (offset) < bandwidth
#define event_at_offset(bi, offset) band_lower_left[(bi)].event_idx - (offset)
#define kmer_at_offset(bi, offset) band_lower_left[(bi)].kmer_idx + (offset)

// #define move_down(curr_band)                                                   \
//   { curr_band.event_idx + 1, curr_band.kmer_idx }
// #define move_right(curr_band)                                                  \
//   { curr_band.event_idx, curr_band.kmer_idx + 1 }

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

#define BAND_ARRAY(r, c) (bands[((r) * (ALN_BANDWIDTH) + (c))])
#define TRACE_ARRAY(r, c) (trace[((r) * (ALN_BANDWIDTH) + (c))])

#define FROM_D 0
#define FROM_U 1
#define FROM_L 2

#define max_gap_threshold 50
#define bandwidth ALN_BANDWIDTH
#define half_bandwidth ALN_BANDWIDTH / 2

#ifndef ALIGN_KERNEL_FLOAT
#define min_average_log_emission -5.0
#define epsilon 1e-10
#else
#define min_average_log_emission -5.0f
#define epsilon 1e-10f
#endif

// inline EventKmerPair move_down(EventKmerPair curr_band) {
//   EventKmerPair ret = {curr_band.event_idx + 1, curr_band.kmer_idx};
//   return ret;
// }
// inline EventKmerPair move_right(EventKmerPair curr_band) {
//   EventKmerPair ret = {curr_band.event_idx, curr_band.kmer_idx + 1};
//   return ret;
// }

/************** Kernels with 2D thread models **************/

//******************************************************************************************************
/*pre kernel*/
//******************************************************************************************************
__attribute__((num_compute_units(1)))
__attribute__((reqd_work_group_size(128, 1, 1))) __kernel void
align_kernel_pre_2d(
    volatile __global char *restrict read, __global int32_t *restrict read_len,
    __global ptr_t *restrict read_ptr, __global ptr_t *restrict event_ptr,
    volatile __global model_t *restrict models, int32_t n_bam_rec,
    volatile __global model_t *restrict model_kmer_caches,
    volatile __global float *restrict bands1,
    volatile __global uint8_t *restrict trace1,
    volatile __global EventKmerPair *restrict band_lower_left1) {
  //   printf("Kernel called\n");
  // CUDA
  // int i = blockDim.y * blockIdx.y + threadIdx.y;
  // int tid=blockIdx.x*blockDim.x+threadIdx.x;

  size_t i = get_global_id(1);
  size_t tid = get_global_id(0);

  if (i < n_bam_rec) {
    volatile __global char *sequence = &read[read_ptr[i]];
    int32_t sequence_len = read_len[i];
    // int32_t n_event = n_events[i];
    volatile __global model_t *model_kmer_cache =
        &model_kmer_caches[read_ptr[i]];
    volatile __global float *bands =
        &bands1[(read_ptr[i] + event_ptr[i]) * ALN_BANDWIDTH];
    volatile __global uint8_t *trace =
        &trace1[(read_ptr[i] + event_ptr[i]) * ALN_BANDWIDTH];
    volatile __global EventKmerPair *band_lower_left =
        &band_lower_left1[read_ptr[i] + event_ptr[i]];

    // int32_t n_events = n_event;
    int32_t n_kmers = sequence_len - KMER_SIZE + 1;
// fprintf(stderr,"n_kmers : %d\n",n_kmers);

// transition penalties
// float events_per_kmer = (float)n_events / n_kmers;
// float p_stay = 1 - (1 / (events_per_kmer + 1));

// setting a tiny skip penalty helps keep the true alignment within the adaptive
// band this was empirically determined
// double epsilon = 1e-10;
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
                    // #pragma unroll
      for (int32_t i = 0; i < n_kmers; ++i) {
        // kmer_ranks[i] = get_kmer_rank(substring, KMER_SIZE);
        volatile __global char *substring = &sequence[i];

        uint32_t kmer_ranks = get_kmer_rank(substring, KMER_SIZE);
        model_kmer_cache[i] = models[kmer_ranks];
      }
    }

    if (tid < bandwidth) {
#pragma unroll
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
