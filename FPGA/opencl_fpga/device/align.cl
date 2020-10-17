#include "k_structs.h"
//#include "f5c.h"

// __kernel void align(__global struct twoArrays *arg1) {
//   int gid = get_global_id(0);

//   arg1->result[gid] = arg1->a[gid] + arg1->b[gid];
// }

__kernel void align(__global  AlignedPair *out_2, __global char *sequence,
                    int sequence_len, __global event_table* events,
                    __global model_t *models, __global scalings_t* scaling,
                    float sample_rate, long n_event_align_pairs) {

  int gid = get_global_id(0);
  // arg1->result[gid] = arg1->a[gid] + arg1->b[gid];

//     size_t strand_idx = 0;
//     size_t k = 6;

//     // size_t n_events = events[strand_idx].n;
//     size_t n_events = events.n;
//     size_t n_kmers = sequence_len - k + 1;
//     //fprintf(stderr,"n_kmers : %d\n",n_kmers);
//     // backtrack markers
//     const uchar FROM_D = 0;
//     const uchar FROM_U = 1;
//     const uchar FROM_L = 2;

//     // qc
//     double min_average_log_emission = -5.0;
//     int max_gap_threshold = 50;

//     // banding
//     int bandwidth = ALN_BANDWIDTH;
//     int half_bandwidth = ALN_BANDWIDTH / 2;

//     // transition penalties
//     double events_per_kmer = (double)n_events / n_kmers;
//     double p_stay = 1 - (1 / (events_per_kmer + 1));

//     // setting a tiny skip penalty helps keep the true alignment within the adaptive band
//     // this was empirically determined
//     double epsilon = 1e-10;
//     double lp_skip = log(epsilon);
//     double lp_stay = log(p_stay);
//     double lp_step = log(1.0 - exp(lp_skip) - exp(lp_stay));
//     double lp_trim = log(0.01);

//     // dp matrix
//     size_t n_rows = n_events + 1;
//     size_t n_cols = n_kmers + 1;
//     size_t n_bands = n_rows + n_cols;

//     // Initialize

//     // Precompute k-mer ranks to avoid doing this in the inner loop
//     size_t* kmer_ranks = (size_t*)malloc(sizeof(size_t) * n_kmers);
//     MALLOC_CHK(kmer_ranks);

//     for (size_t i = 0; i < n_kmers; ++i) {
//         //>>>>>>>>> New replacement begin
//         char* substring = &sequence[i];
//         kmer_ranks[i] = get_kmer_rank(substring, k);
//         //<<<<<<<<< New replacement over
//     }

// #ifdef ALIGN_2D_ARRAY
//     float** bands = (float**)malloc(sizeof(float*) * n_bands);
//     MALLOC_CHK(bands);
//     uchar** trace = (uchar**)malloc(sizeof(uchar*) * n_bands);
//     MALLOC_CHK(trace);
// #else
//     float* bands = (float*)malloc(sizeof(float) * n_bands * bandwidth);
//     MALLOC_CHK(bands);
//     uchar* trace = (uchar*)malloc(sizeof(uchar) * n_bands * bandwidth);
//     MALLOC_CHK(trace);
// #endif
//     for (size_t i = 0; i < n_bands; i++) {
//     #ifdef ALIGN_2D_ARRAY
//         bands[i] = (float*)malloc(sizeof(float) * bandwidth);
//         MALLOC_CHK(bands[i]);
//         trace[i] = (uchar*)malloc(sizeof(uchar) * bandwidth);
//         MALLOC_CHK(trace[i]);
//     #endif

//         for (int j = 0; j < bandwidth; j++) {
//             BAND_ARRAY(i,j) = -INFINITY;
//             TRACE_ARRAY(i,j) = 0;
//         }
//     }

//     // Keep track of the event/kmer index for the lower left corner of the band
//     // these indices are updated at every iteration to perform the adaptive banding
//     // Only the first two  have their coordinates initialized, the rest are computed adaptively

//     struct EventKmerPair {
//         int event_idx;
//         int kmer_idx;
//     };
//     //>>>>>>>>>>>>>>>>>New Replacement Begin
//     struct EventKmerPair* band_lower_left =
//         (struct EventKmerPair*)malloc(sizeof(struct EventKmerPair) * n_bands);
//     MALLOC_CHK(band_lower_left);
//     //std::vector<EventKmerPair> band_lower_left(n_);
//     //<<<<<<<<<<<<<<<<<New Replacement over

//     // initialize range of first two
//     band_lower_left[0].event_idx = half_bandwidth - 1;
//     band_lower_left[0].kmer_idx = -1 - half_bandwidth;
//     band_lower_left[1] = move_down(band_lower_left[0]);

//     int start_cell_offset = band_kmer_to_offset(0, -1);
//     assert(is_offset_valid(start_cell_offset));
//     assert(band_event_to_offset(0, -1) == start_cell_offset);
//     BAND_ARRAY(0,start_cell_offset) = 0.0f;

//     // band 1: first event is trimmed
//     int first_trim_offset = band_event_to_offset(1, 0);
//     assert(kmer_at_offset(1, first_trim_offset) == -1);
//     assert(is_offset_valid(first_trim_offset));
//     BAND_ARRAY(1,first_trim_offset) = lp_trim;
//     TRACE_ARRAY(1,first_trim_offset) = FROM_U;

//     int fills = 0;
// #ifdef DEBUG_ADAPTIVE
//     fprintf(stderr, "[trim] bi: %d o: %d e: %d k: %d s: %.2lf\n", 1,
//             first_trim_offset, 0, -1, bands[1][first_trim_offset]);
// #endif

//     // fill in remaining bands
//     for (size_t band_idx = 2; band_idx < n_bands; ++band_idx) {
//         // Determine placement of this band according to Suzuki's adaptive algorithm
//         // When both ll and ur are out-of-band (ob) we alternate movements
//         // otherwise we decide based on scores
//         float ll = BAND_ARRAY(band_idx - 1,0);
//         float ur = BAND_ARRAY(band_idx - 1,bandwidth - 1);
//         bool ll_ob = ll == -INFINITY;
//         bool ur_ob = ur == -INFINITY;

//         bool right = false;
//         if (ll_ob && ur_ob) {
//             right = band_idx % 2 == 1;
//         } else {
//             right = ll < ur; // Suzuki's rule
//         }

//         if (right) {
//             band_lower_left[band_idx] =
//                 move_right(band_lower_left[band_idx - 1]);
//         } else {
//             band_lower_left[band_idx] =
//                 move_down(band_lower_left[band_idx - 1]);
//         }
//         // If the trim state is within the band, fill it in here
//         int trim_offset = band_kmer_to_offset(band_idx, -1);
//         if (is_offset_valid(trim_offset)) {
//             int64_t event_idx = event_at_offset(band_idx, trim_offset);
//             if (event_idx >= 0 && event_idx < (int64_t)n_events) {
//                 BAND_ARRAY(band_idx,trim_offset) = lp_trim * (event_idx + 1);
//                 TRACE_ARRAY(band_idx,trim_offset) = FROM_U;
//             } else {
//                 BAND_ARRAY(band_idx,trim_offset) = -INFINITY;
//             }
//         }

//         // Get the offsets for the first and last event and kmer
//         // We restrict the inner loop to only these values
//         int kmer_min_offset = band_kmer_to_offset(band_idx, 0);
//         int kmer_max_offset = band_kmer_to_offset(band_idx, n_kmers);
//         int event_min_offset = band_event_to_offset(band_idx, n_events - 1);
//         int event_max_offset = band_event_to_offset(band_idx, -1);

//         int min_offset = MAX(kmer_min_offset, event_min_offset);
//         min_offset = MAX(min_offset, 0);

//         int max_offset = MIN(kmer_max_offset, event_max_offset);
//         max_offset = MIN(max_offset, bandwidth);

//         for (int offset = min_offset; offset < max_offset; ++offset) {
//             int event_idx = event_at_offset(band_idx, offset);
//             int kmer_idx = kmer_at_offset(band_idx, offset);

//             size_t kmer_rank = kmer_ranks[kmer_idx];

//             int offset_up = band_event_to_offset(band_idx - 1, event_idx - 1);
//             int offset_left = band_kmer_to_offset(band_idx - 1, kmer_idx - 1);
//             int offset_diag = band_kmer_to_offset(band_idx - 2, kmer_idx - 1);

// #ifdef DEBUG_ADAPTIVE
//             // verify loop conditions
//             assert(kmer_idx >= 0 && kmer_idx < n_kmers);
//             assert(event_idx >= 0 && event_idx < n_events);
//             assert(offset_diag ==
//                    band_event_to_offset(band_idx - 2, event_idx - 1));
//             assert(offset_up - offset_left == 1);
//             assert(offset >= 0 && offset < bandwidth);
// #endif

//             float up = is_offset_valid(offset_up)
//                            ? BAND_ARRAY(band_idx - 1,offset_up)
//                            : -INFINITY;
//             float left = is_offset_valid(offset_left)
//                              ? BAND_ARRAY(band_idx - 1,offset_left)
//                              : -INFINITY;
//             float diag = is_offset_valid(offset_diag)
//                              ? BAND_ARRAY(band_idx - 2,offset_diag)
//                              : -INFINITY;

//             float lp_emission =
//                 log_probability_match_r9(scaling, models, events, event_idx,
//                                          kmer_rank, strand_idx, sample_rate);
//             //fprintf(stderr, "lp emiision : %f , event idx %d, kmer rank %d\n", lp_emission,event_idx,kmer_rank);
//             float score_d = diag + lp_step + lp_emission;
//             float score_u = up + lp_stay + lp_emission;
//             float score_l = left + lp_skip;

//             float max_score = score_d;
//             uchar from = FROM_D;

//             max_score = score_u > max_score ? score_u : max_score;
//             from = max_score == score_u ? FROM_U : from;
//             max_score = score_l > max_score ? score_l : max_score;
//             from = max_score == score_l ? FROM_L : from;

// #ifdef DEBUG_ADAPTIVE
//             fprintf(stderr,
//                     "[adafill] offset-up: %d offset-diag: %d offset-left: %d\n",
//                     offset_up, offset_diag, offset_left);
//             fprintf(stderr, "[adafill] up: %.2lf diag: %.2lf left: %.2lf\n", up,
//                     diag, left);
//             fprintf(stderr,
//                     "[adafill] bi: %d o: %d e: %d k: %d s: %.2lf f: %d emit: "
//                     "%.2lf\n",
//                     band_idx, offset, event_idx, kmer_idx, max_score, from,
//                     lp_emission);
// #endif
//             BAND_ARRAY(band_idx,offset) = max_score;
//             TRACE_ARRAY(band_idx,offset) = from;
//             fills += 1;
//         }
//     }

//     //
//     // Backtrack to compute alignment
//     //
//     double sum_emission = 0;
//     double n_aligned_events = 0;

//     //>>>>>>>>>>>>>> New replacement begin
//     // std::vector<AlignedPair> out;

//     int outIndex = 0;
//     //<<<<<<<<<<<<<<<<New Replacement over

//     float max_score = -INFINITY;
//     int curr_event_idx = 0;
//     int curr_kmer_idx = n_kmers - 1;

//     // Find best score between an event and the last k-mer. after trimming the remaining evnets
//     for (size_t event_idx = 0; event_idx < n_events; ++event_idx) {
//         int band_idx = event_kmer_to_band(event_idx, curr_kmer_idx);

//         //>>>>>>>New  replacement begin
//         /*assert(band_idx < bands.size());*/
//         assert((size_t)band_idx < n_bands);
//         //<<<<<<<<New Replacement over
//         int offset = band_event_to_offset(band_idx, event_idx);
//         if (is_offset_valid(offset)) {
//             float s =
//                 BAND_ARRAY(band_idx,offset) + (n_events - event_idx) * lp_trim;
//             if (s > max_score) {
//                 max_score = s;
//                 curr_event_idx = event_idx;
//             }
//         }
//     }

// #ifdef DEBUG_ADAPTIVE
//     fprintf(stderr, "[adaback] ei: %d ki: %d s: %.2f\n", curr_event_idx,
//             curr_kmer_idx, max_score);
// #endif

//     int curr_gap = 0;
//     int max_gap = 0;
//     while (curr_kmer_idx >= 0 && curr_event_idx >= 0) {
//         // emit alignment
//         //>>>>>>>New Repalcement begin
//         assert(outIndex < (int)(n_events * 2));
//         out_2[outIndex].ref_pos = curr_kmer_idx;
//         out_2[outIndex].read_pos = curr_event_idx;
//         outIndex++;
//         // out.push_back({curr_kmer_idx, curr_event_idx});
//         //<<<<<<<<<New Replacement over

// #ifdef DEBUG_ADAPTIVE
//         fprintf(stderr, "[adaback] ei: %d ki: %d\n", curr_event_idx,
//                 curr_kmer_idx);
// #endif
//         // qc stats
//         //>>>>>>>>>>>>>>New Replacement begin
//         char* substring = &sequence[curr_kmer_idx];
//         size_t kmer_rank = get_kmer_rank(substring, k);
//         //<<<<<<<<<<<<<New Replacement over
//         float tempLogProb = log_probability_match_r9(
//             scaling, models, events, curr_event_idx, kmer_rank, 0, sample_rate);

//         sum_emission += tempLogProb;
//         //fprintf(stderr, "lp_emission %f \n", tempLogProb);
//         //fprintf(stderr,"lp_emission %f, sum_emission %f, n_aligned_events %d\n",tempLogProb,sum_emission,outIndex);

//         n_aligned_events += 1;

//         int band_idx = event_kmer_to_band(curr_event_idx, curr_kmer_idx);
//         int offset = band_event_to_offset(band_idx, curr_event_idx);
//         assert(band_kmer_to_offset(band_idx, curr_kmer_idx) == offset);

//         uchar from = TRACE_ARRAY(band_idx,offset);
//         if (from == FROM_D) {
//             curr_kmer_idx -= 1;
//             curr_event_idx -= 1;
//             curr_gap = 0;
//         } else if (from == FROM_U) {
//             curr_event_idx -= 1;
//             curr_gap = 0;
//         } else {
//             curr_kmer_idx -= 1;
//             curr_gap += 1;
//             max_gap = MAX(curr_gap, max_gap);
//         }
//     }

//     //>>>>>>>>New replacement begin
//     // std::reverse(out.begin(), out.end());
//     int c;
//     int end = outIndex - 1;
//     for (c = 0; c < outIndex / 2; c++) {
//         int ref_pos_temp = out_2[c].ref_pos;
//         int read_pos_temp = out_2[c].read_pos;
//         out_2[c].ref_pos = out_2[end].ref_pos;
//         out_2[c].read_pos = out_2[end].read_pos;
//         out_2[end].ref_pos = ref_pos_temp;
//         out_2[end].read_pos = read_pos_temp;
//         end--;
//     }

//     // if(outIndex>1){
//     //   AlignedPair temp={out_2[0].ref_pos,out[0].read_pos};
//     //   int i;
//     //   for(i=0;i<outIndex-1;i++){
//     //     out_2[i]={out_2[outIndex-1-i].ref_pos,out[outIndex-1-i].read_pos};
//     //   }
//     //   out[outIndex-1]={temp.ref_pos,temp.read_pos};
//     // }
//     //<<<<<<<<<New replacement over

//     // QC results
//     double avg_log_emission = sum_emission / n_aligned_events;
//     //fprintf(stderr,"sum_emission %f, n_aligned_events %f, avg_log_emission %f\n",sum_emission,n_aligned_events,avg_log_emission);
//     //>>>>>>>>>>>>>New replacement begin
//     bool spanned = out_2[0].ref_pos == 0 &&
//                    out_2[outIndex - 1].ref_pos == int(n_kmers - 1);
//     // bool spanned = out.front().ref_pos == 0 && out.back().ref_pos == n_kmers - 1;
//     //<<<<<<<<<<<<<New replacement over
//     //bool failed = false;
//     if (avg_log_emission < min_average_log_emission || !spanned ||
//         max_gap > max_gap_threshold) {
//         //failed = true;
//         //>>>>>>>>>>>>>New replacement begin
//         outIndex = 0;
//         // out.clear();
//         //free(out_2);
//         //out_2 = NULL;
//         //<<<<<<<<<<<<<New replacement over
//     }

//     free(kmer_ranks);
// #ifdef ALIGN_2D_ARRAY
//     for (size_t i = 0; i < n_bands; i++) {
//         free(bands[i]);
//         free(trace[i]);
//     }
// #endif
//     free(bands);
//     free(trace);
//     free(band_lower_left);
//     //fprintf(stderr, "ada\t%s\t%s\t%.2lf\t%zu\t%.2lf\t%d\t%d\t%d\n", read.read_name.substr(0, 6).c_str(), failed ? "FAILED" : "OK", events_per_kmer, sequence.size(), avg_log_emission, curr_event_idx, max_gap, fills);
//     //outSize=outIndex;
//     //if(outIndex>500000)fprintf(stderr, "Max outSize %d\n", outIndex);
//     return outIndex;
// }


}

