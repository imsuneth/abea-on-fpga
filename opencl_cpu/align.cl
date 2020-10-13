#include "k_structs.h"
// #include "f5c.h"

// __kernel void align(__global AlignedPair *out_2, __global char *sequence,
//                     int sequence_len, event_table events,
//                     __global model_t *models, scalings_t scaling,
//                     float sample_rate) {

//   int gid = get_global_id(0);
//   // arg1->result[gid] = arg1->a[gid] + arg1->b[gid];
// }

__kernel void align(__global struct twoArrays *arg1) {
  int gid = get_global_id(0);

  arg1->result[gid] = arg1->a[gid] + arg1->b[gid];
}