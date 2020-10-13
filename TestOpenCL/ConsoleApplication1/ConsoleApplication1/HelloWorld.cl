struct twoArrays {
  float a[10];
  float b[10];
  float result[10];
};

__kernel void hello_kernel(__global struct twoArrays *arg1) {

  int gid = get_global_id(0);
  arg1->result[gid] = arg1->a[gid] + arg1->b[gid];
}