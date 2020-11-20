
#include "helper.h"
//******************************************************************************************************
/*post kernel*/
//******************************************************************************************************
__attribute__((reqd_work_group_size(1, 1, 1))) __kernel void align_kernel_post(
    __global AlignedPair *restrict event_align_pairs,
    __global int32_t *restrict n_event_align_pairs,
    __global int32_t *restrict read_len, __global ptr_t *restrict read_ptr,
    __global event1_t *restrict event_table, // There is a built-in event_t.
                                             // Therefore, renamed as event1_t
    __global int32_t *restrict n_events, __global ptr_t *restrict event_ptr,
    int32_t n_bam_rec, __global scalings_t *restrict scalings,

    __global model_t *restrict model_kmer_caches,
    __global float *restrict bands1, __global uint8_t *restrict trace1,
    __global EventKmerPair *restrict band_lower_left1) {
  // CUDA
  // int i = blockDim.y * blockIdx.y + threadIdx.y;
  // int offset=blockIdx.x*blockDim.x+threadIdx.x;

  // size_t i = get_global_id(1);
  // size_t tid = get_global_id(0);

#ifndef WARP_HACK
  // int i = blockDim.x * blockIdx.x + threadIdx.x; //CUDA
  size_t i = get_global_id(0);

  if (i < n_bam_rec) {
#else
  // int tid = blockDim.x * blockIdx.x + threadIdx.x; //CUDA
  size_t tid = get_global_id(0);
  int i = tid / 32;
  if (i < n_bam_rec && tid % 32 == 0) {
#endif
    __global AlignedPair *out_2 = &event_align_pairs[event_ptr[i] * 2];
    int32_t sequence_len = read_len[i];
    __global event1_t *events = &event_table[event_ptr[i]];
    int32_t n_event = n_events[i];
    scalings_t scaling = scalings[i];
    __global model_t *model_kmer_cache = &model_kmer_caches[read_ptr[i]];
    __global float *bands =
        &bands1[(read_ptr[i] + event_ptr[i]) * ALN_BANDWIDTH];
    __global uint8_t *trace =
        &trace1[(read_ptr[i] + event_ptr[i]) * ALN_BANDWIDTH];
    __global EventKmerPair *band_lower_left =
        &band_lower_left1[read_ptr[i] + event_ptr[i]];
    ;

    // fprintf(stderr, "%s\n", sequence);
    // fprintf(stderr, "Scaling %f %f", scaling.scale, scaling.shift);

    // size_t strand_idx = 0;
    // size_t k = 6;

    // size_t n_events = events[strand_idx].n;
    int32_t n_events = n_event;
    int32_t n_kmers = sequence_len - KMER_SIZE + 1;
    // fprintf(stderr,"n_kmers : %d\n",n_kmers);
    // backtrack markers
    // const uint8_t FROM_D = 0;
    // const uint8_t FROM_U = 1;
    // const uint8_t FROM_L = 2;

    // qc
    // double min_average_log_emission = -5.0;
    // int max_gap_threshold = 50;

    // banding
    // int bandwidth = ALN_BANDWIDTH;
    // half_bandwidth = bandwidth / 2;

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
    //
    // Backtrack to compute alignment
    //
    double sum_emission = 0;
    double n_aligned_events = 0;

    //>>>>>>>>>>>>>> New replacement begin
    // std::vector<AlignedPair> out;

    int outIndex = 0;
    //<<<<<<<<<<<<<<<<New Replacement over

    float max_score = -INFINITY;
    int curr_event_idx = 0;
    int curr_kmer_idx = n_kmers - 1;

    // Find best score between an event and the last k-mer. after trimming the
    // remaining evnets
    for (int32_t event_idx = 0; event_idx < n_events; ++event_idx) {
      int band_idx = event_kmer_to_band(event_idx, curr_kmer_idx);

      //>>>>>>>New  replacement begin
      /*assert(band_idx < bands.size());*/

      // assert(band_idx < n_bands);

      //<<<<<<<<New Replacement over
      int offset = band_event_to_offset(band_idx, event_idx);
      if (is_offset_valid(offset)) {
        float s =
            BAND_ARRAY(band_idx, offset) + (n_events - event_idx) * lp_trim;
        if (s > max_score) {
          max_score = s;
          curr_event_idx = event_idx;
        }
      }
    }

#ifdef DEBUG_ADAPTIVE
    fprintf(stderr, "[adaback] ei: %d ki: %d s: %.2f\n", curr_event_idx,
            curr_kmer_idx, max_score);
#endif

    int curr_gap = 0;
    int max_gap = 0;
    while (curr_kmer_idx >= 0 && curr_event_idx >= 0) {
      // emit alignment
      //>>>>>>>New Repalcement begin
      // assert(outIndex < n_events * 2);
      out_2[outIndex].ref_pos = curr_kmer_idx;
      out_2[outIndex].read_pos = curr_event_idx;
      outIndex++;
      // out.push_back({curr_kmer_idx, curr_event_idx});
      //<<<<<<<<<New Replacement over

#ifdef DEBUG_ADAPTIVE
      fprintf(stderr, "[adaback] ei: %d ki: %d\n", curr_event_idx,
              curr_kmer_idx);
#endif
      // qc stats
      //>>>>>>>>>>>>>>New Replacement begin
      // char* substring = &sequence[curr_kmer_idx];
      // int32_t kmer_rank = get_kmer_rank(substring, KMER_SIZE);
      // //<<<<<<<<<<<<<New Replacement over
      // float tempLogProb = log_probability_match_r9(
      //     scaling, models, events, curr_event_idx, kmer_rank);

#ifndef PROFILE
      float tempLogProb = log_probability_match_r9(
          scaling, model_kmer_cache, events, curr_event_idx, curr_kmer_idx);
      // fprintf(stderr, "lp emiision : %f , event idx %d, kmer rank %d\n",
      // lp_emission,event_idx,kmer_rank);
#else
      float unscaledLevel = events[curr_event_idx].mean;
      float scaledLevel = unscaledLevel;
      model_t model = model_kmer_cache[curr_kmer_idx];
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
      float tempLogProb = log_inv_sqrt_2pi - gp_log_stdv + (-0.5f * a * a);

#endif

      sum_emission += tempLogProb;
      // fprintf(stderr, "lp_emission %f \n", tempLogProb);
      // fprintf(stderr,"lp_emission %f, sum_emission %f, n_aligned_events
      // %d\n",tempLogProb,sum_emission,outIndex);

      n_aligned_events += 1;

      int band_idx = event_kmer_to_band(curr_event_idx, curr_kmer_idx);
      int offset = band_event_to_offset(band_idx, curr_event_idx);
      // assert(band_kmer_to_offset(band_idx, curr_kmer_idx) == offset);

      uint8_t from = TRACE_ARRAY(band_idx, offset);
      if (from == FROM_D) {
        curr_kmer_idx -= 1;
        curr_event_idx -= 1;
        curr_gap = 0;
      } else if (from == FROM_U) {
        curr_event_idx -= 1;
        curr_gap = 0;
      } else {
        curr_kmer_idx -= 1;
        curr_gap += 1;
        max_gap = MAX(curr_gap, max_gap);
      }
    }

#ifndef REVERSAL_ON_CPU
    //>>>>>>>>New replacement begin
    // std::reverse(out.begin(), out.end());
    int c;
    int end = outIndex - 1;
    for (c = 0; c < outIndex / 2; c++) {
      int ref_pos_temp = out_2[c].ref_pos;
      int read_pos_temp = out_2[c].read_pos;
      out_2[c].ref_pos = out_2[end].ref_pos;
      out_2[c].read_pos = out_2[end].read_pos;
      out_2[end].ref_pos = ref_pos_temp;
      out_2[end].read_pos = read_pos_temp;
      end--;
    }

    // if(outIndex>1){
    //   AlignedPair temp={out_2[0].ref_pos,out[0].read_pos};
    //   int i;
    //   for(i=0;i<outIndex-1;i++){
    //     out_2[i]={out_2[outIndex-1-i].ref_pos,out[outIndex-1-i].read_pos};
    //   }
    //   out[outIndex-1]={temp.ref_pos,temp.read_pos};
    // }
    //<<<<<<<<<New replacement over

    //>>>>>>>>>>>>>New replacement begin
    bool spanned = out_2[0].ref_pos == 0 &&
                   out_2[outIndex - 1].ref_pos == (int)(n_kmers - 1);

    // assert(spanned==spanned_before_rev);
    // bool spanned = out.front().ref_pos == 0 && out.back().ref_pos == n_kmers
    // - 1;
    //<<<<<<<<<<<<<New replacement over
#else
    bool spanned = out_2[outIndex - 1].ref_pos == 0 &&
                   out_2[0].ref_pos == (int)(n_kmers - 1);
#endif
    // QC results
    double avg_log_emission = sum_emission / n_aligned_events;
    // fprintf(stderr,"sum_emission %f, n_aligned_events %f, avg_log_emission
    // %f\n",sum_emission,n_aligned_events,avg_log_emission);

    // bool failed = false;
    if (avg_log_emission < min_average_log_emission || !spanned ||
        max_gap > max_gap_threshold) {
      // failed = true;
      //>>>>>>>>>>>>>New replacement begin
      outIndex = 0;
      // out.clear();
      // free(out_2);
      // out_2 = NULL;
      //<<<<<<<<<<<<<New replacement over
    }

    // free(kmer_ranks);
    // for (size_t i = 0; i < n_bands; i++) {
    //     free(bands[i]);
    //     free(trace[i]);
    // }
    // free(bands);
    // free(trace);
    // free(band_lower_left);
    // fprintf(stderr, "ada\t%s\t%s\t%.2lf\t%zu\t%.2lf\t%d\t%d\t%d\n",
    // read.read_name.substr(0, 6).c_str(), failed ? "FAILED" : "OK",
    // events_per_kmer, sequence.size(), avg_log_emission, curr_event_idx,
    // max_gap, fills); outSize=outIndex; if(outIndex>500000)fprintf(stderr,
    // "Max outSize %d\n", outIndex);
    n_event_align_pairs[i] = outIndex;
  }
}
