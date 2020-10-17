#include "f5c.h"

//todo : can make more efficient using bit encoding
static inline uint32_t get_rank(char base) {
    if (base == 'A') { //todo: do we neeed simple alpha?
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
static inline uint32_t get_kmer_rank(const char* str, uint32_t k) {
    //uint32_t p = 1;
    uint32_t r = 0;

    // from last base to first
    for (uint32_t i = 0; i < k; ++i) {
        //r += rank(str[k - i - 1]) * p;
        //p *= size();
        r += get_rank(str[k - i - 1]) << (i << 1);
    }
    return r;
}

//copy a kmer from a reference
static inline void kmer_cpy(char* dest, char* src, uint32_t k) {
    uint32_t i = 0;
    for (i = 0; i < k; i++) {
        dest[i] = src[i];
    }
    dest[i] = '\0';
}

scalings_t estimate_scalings_using_mom(char* sequence, int32_t sequence_len,
                                       model_t* pore_model, event_table et) {
    scalings_t out;
    int32_t n_kmers =
        sequence_len - KMER_SIZE + 1; //todo :strlen can be pre-calculated

    //const Alphabet* alphabet = pore_model.pmalphabet;

    // Calculate summary statistics over the events and
    // the model implied by the read
    double event_level_sum = 0.0f; //do we need double?
    for (size_t i = 0; i < et.n; ++i) {
        event_level_sum += et.event[i].mean;
    }

    double kmer_level_sum = 0.0f;
    double kmer_level_sq_sum = 0.0f;
    for (int32_t i = 0; i < n_kmers; ++i) {
        int32_t kr = get_kmer_rank(&sequence[i], KMER_SIZE);
        double l = pore_model[kr].level_mean;
        //fprintf(stderr,"Kmer : %c%c%c%c%c%c, kmer_rank : %d , kmer_mean : %f \n",sequence[i],sequence[i+1],sequence[i+2],sequence[i+3],sequence[i+4],sequence[i+5],kr,l);
        kmer_level_sum += l;
        kmer_level_sq_sum += l * l;
    }

    double shift = event_level_sum / et.n - kmer_level_sum / n_kmers;

    // estimate scale
    double event_level_sq_sum = 0.0f;
    for (size_t i = 0; i < et.n; ++i) {
        event_level_sq_sum +=
            (et.event[i].mean - shift) * (et.event[i].mean - shift);
    }

    double scale = (event_level_sq_sum / et.n) / (kmer_level_sq_sum / n_kmers);

    //out.set4(shift, scale, 0.0, 1.0);
    out.shift = (float)shift;
    out.scale = (float)scale;

#ifdef DEBUG_ESTIMATED_SCALING
    fprintf(stderr, "event mean: %.2lf kmer mean: %.2lf shift: %.2lf\n",
            event_level_sum / et.n, kmer_level_sum / n_kmers, out.shift);
    fprintf(stderr, "event sq-mean: %.2lf kmer sq-mean: %.2lf scale: %.2lf\n",
            event_level_sq_sum / et.n, kmer_level_sq_sum / n_kmers, out.scale);
    //fprintf(stderr, "truth shift: %.2lf scale: %.2lf\n", pore_model.shift, pore_model.scale);
#endif
    return out;
}

static inline float log_normal_pdf(float x, float gp_mean, float gp_stdv,
                                   float gp_log_stdv) {
    /*INCOMPLETE*/
    float log_inv_sqrt_2pi = -0.918938f; // Natural logarithm
    float a = (x - gp_mean) / gp_stdv;
    return log_inv_sqrt_2pi - gp_log_stdv + (-0.5f * a * a);
    // return 1;
}

static inline float log_probability_match_r9(scalings_t scaling,
                                             model_t* models,
                                             event_table events, int event_idx,
                                             uint32_t kmer_rank, uint8_t strand,
                                             float sample_rate) {
    // event level mean, scaled with the drift value
    strand = 0;
    // assert(kmer_rank < 4096);
    //float level = read.get_drift_scaled_level(event_idx, strand);

    //float time =
    //    (events.event[event_idx].start - events.event[0].start) / sample_rate;
    float unscaledLevel = events.event[event_idx].mean;
    float scaledLevel = unscaledLevel;
    //float scaledLevel = unscaledLevel - time * scaling.shift;

    //fprintf(stderr, "level %f\n",scaledLevel);
    //GaussianParameters gp = read.get_scaled_gaussian_from_pore_model_state(pore_model, strand, kmer_rank);
    float gp_mean =
        scaling.scale * models[kmer_rank].level_mean + scaling.shift;
    float gp_stdv = models[kmer_rank].level_stdv * 1; //scaling.var = 1;
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

#define move_down(curr_band)                                                   \
    { curr_band.event_idx + 1, curr_band.kmer_idx }
#define move_right(curr_band)                                                  \
    { curr_band.event_idx, curr_band.kmer_idx + 1 }

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))


#ifdef ALIGN_2D_ARRAY
    #define BAND_ARRAY(r, c) ( bands[(r)][(c)] )
    #define TRACE_ARRAY(r, c) ( trace[(r)][(c)] )
#else
    #define BAND_ARRAY(r, c) ( bands[((r)*(ALN_BANDWIDTH)+(c))] )
    #define TRACE_ARRAY(r, c) ( trace[((r)*(ALN_BANDWIDTH)+(c))] )
#endif

/************** Kernels with 2D thread models **************/

__kernel void align(
    __global char* read,
    __global int32_t* read_len, 
    __global ptr_t* read_ptr,
    __global int32_t* n_events,
    __global ptr_t* event_ptr, 
    __global model_t* models,
    int32_t n_bam_rec,
    __global model_t* model_kmer_caches,
    __global float *bands1,
    __global uint8_t *trace1, 
    __global EventKmerPair* band_lower_left1
  ){
    
  }