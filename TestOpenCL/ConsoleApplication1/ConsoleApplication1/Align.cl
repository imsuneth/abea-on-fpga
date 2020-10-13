/*
align() in C

int32_t align(AlignedPair *out_2, char *sequence, int32_t sequence_len,
              event_table events, model_t *models, scalings_t scaling,
              float sample_rate)



*/

// from nanopolish
typedef struct {
  int ref_pos;
  int read_pos;
} AlignedPair;

__kernel void align(__global const float *a, __global const float *b,
                    __global float *result) {
  int gid = get_global_id(0);

  result[gid] = a[gid] + b[gid];
}