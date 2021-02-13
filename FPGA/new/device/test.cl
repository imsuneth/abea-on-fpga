#define SIZE 10

__attribute__((reqd_work_group_size(1, 1, 1))) __kernel void
test(__global int *restrict A, __global int *restrict B,
     __global int *restrict C) {

  for (int i = 1; i < SIZE; i++) {
    C[i] = C[i - 1] + B[i - 1];
  }
}