
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

#define PROFILE 1

#define band_event_to_offset_shm(bi, ei)                                       \
  band_lower_left_shm[bi].event_idx - (ei)
#define band_kmer_to_offset_shm(bi, ki) (ki) - band_lower_left_shm[bi].kmer_idx

#define event_at_offset_shm(bi, offset)                                        \
  band_lower_left_shm[(bi)].event_idx - (offset)
#define kmer_at_offset_shm(bi, offset)                                         \
  band_lower_left_shm[(bi)].kmer_idx + (offset)

#define BAND_ARRAY_SHM(r, c) (bands_shm[(r)][(c)])
// #define BAND_ARRAY_SHM(r, c) (bands_shm[(c)][(r)])
//******************************************************************************************************
/*core kernel*/
//******************************************************************************************************
__attribute__((num_compute_units(1))) __attribute__((num_simd_work_items(1)))
__attribute__((reqd_work_group_size(128, 1, 1))) __kernel void
align_kernel_core_2d_shm(
    __global int32_t *restrict read_len, __global ptr_t *restrict read_ptr,
    volatile __global event1_t *restrict
        event_table, // There is a built-in event_t.
                     // Therefore, renamed as event1_t
    __global int32_t *restrict n_events1, __global ptr_t *restrict event_ptr,
    __global scalings_t *restrict scalings, int32_t n_bam_rec,
    volatile __global model_t *restrict model_kmer_caches,
    volatile __global float *restrict band,
    volatile __global uint8_t *restrict traces,
    volatile __global EventKmerPair *restrict band_lower_lefts) {
  //
  // printf("IN CORE KERNEL!\n");
  // CUDA
  // int i = blockDim.y * blockIdx.y + threadIdx.y;
  // int offset=blockIdx.x*blockDim.x+threadIdx.x;

  size_t i = get_global_id(1);
  size_t offset = get_global_id(0);

  // printf("i:%lu, offset:%lu\n", i, offset);

  // if (offset == 0)
  //   printf("Completion:%lu\n", i);

  __local float bands_shm[3][ALN_BANDWIDTH];
  // __local float bands_shm[ALN_BANDWIDTH][3];
  __local EventKmerPair band_lower_left_shm[3];

  // printf("IN CORE KERNEL - IN IF CONDITION!\n");

  int32_t sequence_len = read_len[i];
  volatile __global event1_t *events = &event_table[event_ptr[i]];
  int32_t n_event = n_events1[i];
  scalings_t scaling = scalings[i];
  volatile __global model_t *model_kmer_cache = &model_kmer_caches[read_ptr[i]];
  volatile __global float *bands =
      &band[(read_ptr[i] + event_ptr[i]) * ALN_BANDWIDTH];
  volatile __global uint8_t *trace =
      &traces[(read_ptr[i] + event_ptr[i]) * ALN_BANDWIDTH];
  volatile __global EventKmerPair *band_lower_left =
      &band_lower_lefts[read_ptr[i] + event_ptr[i]];

  // size_t n_events = events[strand_idx].n;
  int32_t n_events = n_event;
  int32_t n_kmers = sequence_len - KMER_SIZE + 1;
  // fprintf(stderr,"n_kmers : %d\n",n_kmers);

  // transition penalties
  float events_per_kmer = (float)n_events / n_kmers;
  float p_stay = 1 - (1 / (events_per_kmer + 1));

  // setting a tiny skip penalty helps keep the true alignment within the
  // adaptive band this was empirically determined
  // double epsilon = 1e-10;

#ifndef ALIGN_KERNEL_FLOAT
  double lp_skip = log(epsilon);
  double lp_stay = log(p_stay);
  double lp_step = log(1.0 - exp(lp_skip) - exp(lp_stay));
  double lp_trim = log(0.01);
#else
  float lp_skip = logf(epsilon);
  float lp_stay = logf(p_stay);
  float lp_step = logf(1.0f - expf(lp_skip) - expf(lp_stay));
  float lp_trim = logf(0.01f);
#endif
  // dp matrix
  int32_t n_rows = n_events + 1;
  int32_t n_cols = n_kmers + 1;
  int32_t n_bands = n_rows + n_cols;

  if (i < n_bam_rec && offset < ALN_BANDWIDTH) {
    BAND_ARRAY_SHM(0, offset) = BAND_ARRAY(2, offset);
    BAND_ARRAY_SHM(1, offset) = BAND_ARRAY(1, offset);
    BAND_ARRAY_SHM(2, offset) = BAND_ARRAY(0, offset);

    band_lower_left_shm[0] = band_lower_left[2];
    band_lower_left_shm[1] = band_lower_left[1];
    band_lower_left_shm[2] = band_lower_left[0];
  }
  // __syncthreads(); //CUDA
  // printf("IN CORE KERNEL - IN LOOP - before barrier 0!\n");
  barrier(CLK_LOCAL_MEM_FENCE); // OpenCL
  // printf("IN CORE KERNEL - IN LOOP - after barrier 0!\n");

  // printf("IN CORE KERNEL - Before loop!\n");
  // fill in remaining bands
  //#pragma unroll
  for (int32_t band_idx = 2; band_idx < n_bands; ++band_idx) {

    // priclntf("Core Kernel - i: %lu, band_idx: %d\n", i, band_idx);

    // printf("IN CORE KERNEL - In loop!\n");
    if (i < n_bam_rec && offset < ALN_BANDWIDTH) {
      if (offset == 0) {
        // Determine placement of this band according to Suzuki's adaptive
        // algorithm When both ll and ur are out-of-band (ob) we alternate
        // movements otherwise we decide based on scores
        // float ll = BAND_ARRAY((band_idx - 1), 0);
        float ll = BAND_ARRAY_SHM((1), 0);
        // float ur = BAND_ARRAY((band_idx - 1),(bandwidth - 1));
        float ur = BAND_ARRAY_SHM((1), (bandwidth - 1));
        bool ll_ob = ll == -INFINITY;
        bool ur_ob = ur == -INFINITY;

        bool right = false;
        if (ll_ob && ur_ob) {
          right = band_idx % 2 == 1;
        } else {
          right = ll < ur; // Suzuki's rule
        }

        if (right) {
          // band_lower_left[band_idx] = band_lower_left_shm[0] =
          //     move_right(band_lower_left_shm[1]);

          band_lower_left[band_idx].kmer_idx = band_lower_left_shm[0].kmer_idx =
              band_lower_left_shm[1].kmer_idx + 1;
          band_lower_left[band_idx].event_idx =
              band_lower_left_shm[0].event_idx =
                  band_lower_left_shm[1].event_idx;
        } else {
          // band_lower_left[band_idx] = band_lower_left_shm[0] =
          //     move_down(band_lower_left_shm[1]);

          band_lower_left[band_idx].event_idx =
              band_lower_left_shm[0].event_idx =
                  band_lower_left_shm[1].event_idx + 1;
          band_lower_left[band_idx].kmer_idx = band_lower_left_shm[0].kmer_idx =
              band_lower_left_shm[1].kmer_idx;
        }
        // If the trim state is within the band, fill it in here
        int trim_offset = band_kmer_to_offset_shm(0, -1);
        if (is_offset_valid(trim_offset)) {
          int32_t event_idx = event_at_offset_shm(0, trim_offset);
          if (event_idx >= 0 && event_idx < n_events) {
            // BAND_ARRAY(band_idx,trim_offset) = lp_trim * (event_idx + 1);
            BAND_ARRAY_SHM(0, trim_offset) = lp_trim * (event_idx + 1);
            TRACE_ARRAY(band_idx, trim_offset) = FROM_U;
          } else {
            // BAND_ARRAY(band_idx,trim_offset) = -INFINITY;
            BAND_ARRAY_SHM(0, trim_offset) = -INFINITY;
          }
        }
      }
    }
    // __syncthreads();
    // printf("IN CORE KERNEL - IN LOOP - before barrier 1!\n");
    barrier(CLK_LOCAL_MEM_FENCE); // OpenCL
    // printf("IN CORE KERNEL - IN LOOP - after barrier 1!\n");

    int kmer_min_offset;
    int kmer_max_offset;
    int event_min_offset;
    int event_max_offset;
    int min_offset;
    int max_offset;
    if (i < n_bam_rec && offset < ALN_BANDWIDTH) {
      // Get the offsets for the first and last event and kmer
      // We restrict the inner loop to only these values
      kmer_min_offset = band_kmer_to_offset_shm(0, 0);
      kmer_max_offset = band_kmer_to_offset_shm(0, n_kmers);
      event_min_offset = band_event_to_offset_shm(0, n_events - 1);
      event_max_offset = band_event_to_offset_shm(0, -1);

      min_offset = MAX(kmer_min_offset, event_min_offset);
      min_offset = MAX(min_offset, 0);

      max_offset = MIN(kmer_max_offset, event_max_offset);
      max_offset = MIN(max_offset, bandwidth);
    }
    // __syncthreads();
    // printf("IN CORE KERNEL - IN LOOP - before barrier 2!\n");
    barrier(CLK_LOCAL_MEM_FENCE); // OpenCL
    // printf("IN CORE KERNEL - IN LOOP - after barrier 2!\n");
    if (i < n_bam_rec && offset < ALN_BANDWIDTH) {
      if (offset >= min_offset && offset < max_offset) {

        int event_idx = event_at_offset_shm(0, offset);
        int kmer_idx = kmer_at_offset_shm(0, offset);

        // int32_t kmer_rank = kmer_ranks[kmer_idx];

        int offset_up = band_event_to_offset_shm(1, event_idx - 1);
        int offset_left = band_kmer_to_offset_shm(1, kmer_idx - 1);
        int offset_diag = band_kmer_to_offset_shm(2, kmer_idx - 1);

#ifdef DEBUG_ADAPTIVE
        // verify loop conditions
        assert(kmer_idx >= 0 && kmer_idx < n_kmers);
        assert(event_idx >= 0 && event_idx < n_events);
        assert(offset_diag == band_event_to_offset_shm(2, event_idx - 1));
        assert(offset_up - offset_left == 1);
        assert(offset >= 0 && offset < bandwidth);
#endif // DEBUG_ADAPTIVE

        float up = is_offset_valid(offset_up) ? BAND_ARRAY_SHM(1, offset_up)
                                              : -INFINITY;
        float left = is_offset_valid(offset_left)
                         ? BAND_ARRAY_SHM(1, offset_left)
                         : -INFINITY;
        float diag = is_offset_valid(offset_diag)
                         ? BAND_ARRAY_SHM(2, offset_diag)
                         : -INFINITY;

#ifndef PROFILE
        float lp_emission = log_probability_match_r9(
            scaling, model_kmer_cache, events, event_idx, kmer_idx);
        // fprintf(stderr, "lp emiision : %f , event idx %d, kmer rank %d\n",
        // lp_emission,event_idx,kmer_rank);
#else
        float unscaledLevel = events[event_idx].mean;
        float scaledLevel = unscaledLevel;
        model_t model = model_kmer_cache[kmer_idx];
        float gp_mean = scaling.scale * model.level_mean + scaling.shift;
        float gp_stdv = model.level_stdv; // scaling.var = 1;

#ifdef CACHED_LOG
        float gp_log_stdv = model.level_log_stdv;
#else
#ifndef ALIGN_KERNEL_FLOAT
        float gp_log_stdv = log(gp_stdv); // scaling.log_var = log(1)=0;
#else
        float gp_log_stdv = logf(gp_stdv); // scaling.log_var = log(1)=0;
#endif
#endif

        float a = (scaledLevel - gp_mean) / gp_stdv;
        float lp_emission = log_inv_sqrt_2pi - gp_log_stdv + (-0.5f * a * a);

#endif

        float score_d = diag + lp_step + lp_emission;
        float score_u = up + lp_stay + lp_emission;
        float score_l = left + lp_skip;

        float max_score = score_d;
        uint8_t from = FROM_D;

        max_score = score_u > max_score ? score_u : max_score;
        from = max_score == score_u ? FROM_U : from;
        max_score = score_l > max_score ? score_l : max_score;
        from = max_score == score_l ? FROM_L : from;

#ifdef DEBUG_ADAPTIVE
        fprintf(stderr,
                "[adafill] offset-up: %d offset-diag: %d offset-left: %d\n",
                offset_up, offset_diag, offset_left);
        fprintf(stderr, "[adafill] up: %.2lf diag: %.2lf left: %.2lf\n", up,
                diag, left);
        fprintf(stderr,
                "[adafill] bi: %d o: %d e: %d k: %d s: %.2lf f: %d emit: "
                "%.2lf\n",
                band_idx, offset, event_idx, kmer_idx, max_score, from,
                lp_emission);
#endif // DEBUG_ADAPTIVE \
       // BAND_ARRAY(band_idx,offset) = max_score;
        BAND_ARRAY_SHM(0, offset) = max_score;
        TRACE_ARRAY(band_idx, offset) = from;
        // fills += 1;
      }
    }
    // __syncthreads();
    // printf("IN CORE KERNEL - IN LOOP - before barrier 3!\n");
    barrier(CLK_LOCAL_MEM_FENCE); // OpenCL
    // printf("IN CORE KERNEL - IN LOOP - after barrier 3!\n");

    if (i < n_bam_rec && offset < ALN_BANDWIDTH) {
      BAND_ARRAY(band_idx, offset) = BAND_ARRAY_SHM(0, offset);

      BAND_ARRAY_SHM(2, offset) = BAND_ARRAY_SHM(1, offset);
      BAND_ARRAY_SHM(1, offset) = BAND_ARRAY_SHM(0, offset);
      BAND_ARRAY_SHM(0, offset) = -INFINITY;

      if (offset == 0) {
        band_lower_left_shm[2] = band_lower_left_shm[1];
        band_lower_left_shm[1] = band_lower_left_shm[0];
      }
    }
    // __syncthreads();
    // printf("IN CORE KERNEL - IN LOOP - before barrier 4!\n");
    barrier(CLK_LOCAL_MEM_FENCE); // OpenCL
    // printf("IN CORE KERNEL - IN LOOP - after barrier 4!\n");

    // printf("IN CORE KERNEL - After loop!\n");
  }

  // else {

  //   // printf("Deviated work item ");
  //   // printf("i:%lu, offset:%lu\n", i, offset);
  //   int32_t sequence_len = read_len[i];
  //   int32_t n_event = n_events1[i];

  //   int32_t n_events = n_event;
  //   int32_t n_kmers = sequence_len - KMER_SIZE + 1;

  //   // dp matrix
  //   int32_t n_rows = n_events + 1;
  //   int32_t n_cols = n_kmers + 1;
  //   int32_t n_bands = n_rows + n_cols;
  //   barrier(CLK_LOCAL_MEM_FENCE);
  //   for (int32_t band_idx = 2; band_idx < n_bands; ++band_idx) {
  //     barrier(CLK_LOCAL_MEM_FENCE);
  //     barrier(CLK_LOCAL_MEM_FENCE);
  //     barrier(CLK_LOCAL_MEM_FENCE);
  //     barrier(CLK_LOCAL_MEM_FENCE);
  //   }
  // }
}
