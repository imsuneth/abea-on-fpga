
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

#define max_gap_threshold 50
#define min_average_log_emission -5.0
#define epsilon 1e-10

#define FROM_D 0
#define FROM_U 1
#define FROM_L 2

inline float log_normal_pdf(float x, float gp_mean, float gp_stdv,
                            float gp_log_stdv) {
  /*INCOMPLETE*/
  float log_inv_sqrt_2pi = -0.918938f; // Natural logarithm
  float a = (x - gp_mean) / gp_stdv;
  return log_inv_sqrt_2pi - gp_log_stdv + (-0.5f * a * a);
  // return 1;
}

inline float log_probability_match_r9(scalings_t scaling,
                                      __global model_t *restrict models,
                                      __global event1_t *restrict events,
                                      int event_idx, uint32_t kmer_rank) {
  // event level mean, scaled with the drift value
  // strand = 0;
  // assert(kmer_rank < 4096);
  // float level = read.get_drift_scaled_level(event_idx, strand);

  // float time =
  //    (events.event[event_idx].start - events.event[0].start) / sample_rate;
  // float unscaledLevel = events.event[event_idx].mean;
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

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

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
// __attribute__((num_compute_units(1)))
__attribute__((task)) __kernel void align_kernel_single(
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
  // #pragma ivdep array(read_len)
  // #pragma ivdep array(read_ptr)
  // #pragma ivdep array(event_table)
  // #pragma ivdep array(n_events1)
  // #pragma ivdep array(event_ptr)
  // #pragma ivdep array(scalings)
  // #pragma ivdep array(models)
  // #pragma ivdep array(kmer_ranks1)
  // #pragma ivdep array(band)
  // #pragma ivdep array(traces)
  // #pragma ivdep array(band_lower_lefts)
  for (int32_t ii = 0; ii < n_bam_rec; ii++) {
    // fprintf(stderr, "%s\n", sequence);
    // fprintf(stderr, "Scaling %f %f", scaling.scale, scaling.shift);
    // printf("read:%lu\n", ii);
    // AlignedPair* out_2 = db->event_align_pairs[i];
    // char* sequence = db->read[i];
    // int32_t sequence_len = db->read_len[i];
    // event_table events = db->et[i];
    // model_t* models = core->model;
    // scalings_t scaling = db->scalings[i];
    // float sample_rate = db->f5[i]->sample_rate;

    // __global AlignedPair *out_2 = &event_align_pairs[event_ptr[ii] * 2];
    // __global char *sequence = &read[read_ptr[ii]];
    int32_t sequence_len = read_len[ii];
    // printf("read_len[%lu] = %d\n", ii, read_len[ii]);

    // __global event1_t *events = &event_table[event_ptr[ii]];
    __global event1_t *events = event_table + event_ptr[ii];

    int32_t n_events = n_events1[ii];

    scalings_t scaling = scalings[ii];

    // __global float *bands =
    //     &band[(read_ptr[ii] + event_ptr[ii]) * ALN_BANDWIDTH];
    __global float *bands =
        band + (read_ptr[ii] + event_ptr[ii]) * ALN_BANDWIDTH;

    // __global uint8_t *trace =
    //     &traces[(read_ptr[ii] + event_ptr[ii]) * ALN_BANDWIDTH];
    __global uint8_t *trace =
        traces + (read_ptr[ii] + event_ptr[ii]) * ALN_BANDWIDTH;

    // __global EventKmerPair *band_lower_left =
    //     &band_lower_lefts[read_ptr[ii] + event_ptr[ii]];
    __global EventKmerPair *band_lower_left =
        band_lower_lefts + read_ptr[ii] + event_ptr[ii];

    // __global uint32_t *kmer_ranks = &kmer_ranks1[read_ptr[ii]];
    __global uint32_t *kmer_ranks = kmer_ranks1 + read_ptr[ii];

    // int32_t n_events = n_event; // <------ diff
    int32_t n_kmers = sequence_len - KMER_SIZE + 1;

    // transition penalties
    double events_per_kmer = (double)n_events / n_kmers;
    float p_stay = 1 - (1 / (events_per_kmer + 1));

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

    // printf("n_bands = %lu\n", n_bands);

    // int fills = 0;

    // printf("INNER_LOOP!!!!!!!!!!!!!!!\n");
    // fill in remaining bands
    // printf("n_bands %lu\n", n_bands);
    // #pragma unroll
    bool odd_band_idx = true;
// #pragma ivdep array(events)
// #pragma ivdep array(models)
// #pragma ivdep array(kmer_ranks)
// #pragma ivdep array(traces)
// #pragma unroll 4 // for de5net a7; try higher for arria 10
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
        // band_lower_left[band_idx] = move_right(band_lower_left[band_idx -
        // 1]);

        // band_lower_left[band_idx] = band_lower_left[band_idx - 1];
        // band_lower_left[band_idx].kmer_idx++;

        bbl.kmer_idx++;

        // band_lower_left[band_idx].event_idx =
        //     band_lower_left[band_idx - 1].event_idx;
        // band_lower_left[band_idx].kmer_idx =
        //     band_lower_left[band_idx - 1].kmer_idx + 1;

      } else {
        // band_lower_left[band_idx] = move_down(band_lower_left[band_idx -
        // 1]);

        // band_lower_left[band_idx] = band_lower_left[band_idx - 1];
        // band_lower_left[band_idx].event_idx++;

        bbl.event_idx++;

        // band_lower_left[band_idx].event_idx =
        //     band_lower_left[band_idx - 1].event_idx + 1;
        // band_lower_left[band_idx].kmer_idx =
        //     band_lower_left[band_idx - 1].kmer_idx;
      }
      band_lower_left[band_idx] = bbl;

      int trim_offset = band_kmer_to_offset(band_idx, -1);
      // printf("%d ", trim_offset);
      if (is_offset_valid(trim_offset)) {
        int32_t event_idx = event_at_offset(band_idx, trim_offset);
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
      int kmer_min_offset = 0 - band_lower_left[band_idx].kmer_idx;
      // printf("%d ", kmer_min_offset);
      int kmer_max_offset = band_kmer_to_offset(band_idx, n_kmers);
      // printf("%d ", kmer_max_offset);
      int event_min_offset = band_event_to_offset(band_idx, n_events - 1);
      // printf("%d ", event_min_offset);
      int event_max_offset = band_event_to_offset(band_idx, -1);
      // printf("%d ", event_max_offset);
      int min_offset = MAX(kmer_min_offset, event_min_offset);
      min_offset = MAX(min_offset, 0);

      int max_offset = MIN(kmer_max_offset, event_max_offset);
      max_offset = MIN(max_offset, bandwidth);

// #pragma speculated_iterations 3
// #pragma unroll 2
#pragma ivdep array(bands)
#pragma ivdep array(trace)
#pragma ivdep array(band_lower_left)
#pragma unroll 7 
      for (int offset = 0; offset < ALN_BANDWIDTH; ++offset) {

        if (offset >= min_offset && offset < max_offset) {

          int event_idx = event_at_offset(band_idx, offset);
          int kmer_idx = kmer_at_offset(band_idx, offset);

          int32_t kmer_rank = kmer_ranks[kmer_idx];
          // printf("%ld ", kmer_rank);

          float lp_emission = log_probability_match_r9(scaling, models, events,
                                                       event_idx, kmer_rank);

          int offset_up = band_event_to_offset(band_idx - 1, event_idx - 1);
          int offset_left = band_kmer_to_offset(band_idx - 1, kmer_idx - 1);
          int offset_diag = band_kmer_to_offset(band_idx - 2, kmer_idx - 1);

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