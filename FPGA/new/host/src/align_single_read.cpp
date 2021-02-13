#include "f5c.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <math.h>
#include <cstring>
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"

#include "dump_functions.h"

using namespace aocl_utils;

#include "error.h"
#include "f5cmisc_cu.h"
#include "f5cmisc.h"

int print_results = true;
#define VERBOSITY 1

#define AOCL_ALIGNMENT 64

// #define CPU_GPU_PROC

#define STRING_BUFFER_LEN 1024

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel align_kernel_single = NULL;

static cl_program program = NULL;

cl_int status;

// Function prototypes
bool init();
void cleanup();
static void device_info_ulong(cl_device_id device, cl_device_info param, const char *name);
static void device_info_uint(cl_device_id device, cl_device_info param, const char *name);
static void device_info_bool(cl_device_id device, cl_device_info param, const char *name);
static void device_info_string(cl_device_id device, cl_device_info param, const char *name);
static void display_device_info(cl_device_id device);

static void align_ocl(core_t *core, db_t *db);

double total_kernel_time, total_data_time;

// Entry point.
int main(int argc, char *argv[])
{

  if (!init())
  {
    fprintf(stderr, "init() unsuccessful\n");
    return -1;
  }

  fprintf(stderr, "init() successful\n\n");

  // Load dump files
  const char *dump_dir = argv[1];

  int32_t no_of_batches = load_no_of_batches(dump_dir);
  // int32_t no_of_batches = 1;

  fprintf(stderr, "no_of_batches: %d\n", no_of_batches);
  int32_t batch_no = 0;

  int32_t total_no_of_reads = 0;

  for (batch_no = 0; batch_no < no_of_batches; batch_no++)
  {

    fprintf(stderr, "batch_no: %d/%d\t", batch_no, no_of_batches);

    char batch_dir[50];
    snprintf(batch_dir, sizeof(batch_dir), "%s/%ld", dump_dir, batch_no);

    db_t *db;
    db = (db_t *)malloc(sizeof(db_t));
    // db = new db_t();

    core_t *core;
    core = (core_t *)malloc(sizeof(core_t));
    core->opt.verbosity = VERBOSITY;

    load_n_bam_rec(db, batch_dir);
    // db->n_bam_rec = 5;

    // core->model = (model_t*)malloc(sizeof(model_t)*db->n_bam_rec);
    posix_memalign((void **)&core->model, AOCL_ALIGNMENT, sizeof(model_t) * NUM_KMER);

    load_core(core, batch_dir);

    fprintf(stderr, "reads:\t%ld\n", db->n_bam_rec);
    total_no_of_reads += db->n_bam_rec;

    db->n_event_align_pairs = (int32_t *)malloc(sizeof(int32_t) * db->n_bam_rec);
    db->event_align_pairs = (AlignedPair **)malloc(sizeof(AlignedPair *) * db->n_bam_rec);
    // db->read_len = (int32_t *)malloc(sizeof(int32_t) * db->n_bam_rec);
    posix_memalign((void **)&db->read_len, AOCL_ALIGNMENT, sizeof(int32_t) * db->n_bam_rec);
    db->read = (char **)malloc(sizeof(char *) * db->n_bam_rec);
    db->et = (event_table *)malloc(sizeof(event_table) * db->n_bam_rec);
    // db->scalings = (scalings_t *)malloc(sizeof(scalings_t) * db->n_bam_rec);
    posix_memalign((void **)&db->scalings, AOCL_ALIGNMENT, sizeof(scalings_t) * db->n_bam_rec);
    db->f5 = (fast5_t **)malloc(sizeof(fast5_t *) * db->n_bam_rec);

    db_t *db_out;
    db_out = (db_t *)malloc(sizeof(db_t));
    db_out->n_event_align_pairs = (int32_t *)malloc(sizeof(int32_t) * db->n_bam_rec);
    db_out->event_align_pairs = (AlignedPair **)malloc(sizeof(AlignedPair *) * db->n_bam_rec);

    // printf("db, core initialized\n");

    for (int i = 0; i < db->n_bam_rec; i++)
    {

      load_read_inputs(db, i, batch_dir);
      load_read_outputs(db_out, i, batch_dir);
      // posix_memalign((void **)&db->event_align_pairs[i], AOCL_ALIGNMENT, sizeof(AlignedPair) * db_out->n_event_align_pairs[i]);
      // db->event_align_pairs[i] = (AlignedPair *)malloc(sizeof(AlignedPair) * db_out->n_event_align_pairs[i]);
    }

    align_ocl(core, db);

    if (print_results)
    {
      for (int i = 0; i < db->n_bam_rec; i++)
      {
        // compare with original output
        int32_t n_event_align_pairs = db->n_event_align_pairs[i];
        int32_t n_event_align_pairs_out = db_out->n_event_align_pairs[i];

        if (n_event_align_pairs != n_event_align_pairs_out)
        {
          fprintf(stderr, "%d=\toutput: %d (%d)\tFailed\n", i, n_event_align_pairs, n_event_align_pairs_out);
          //break;
        }
        else
        {
          fprintf(stderr, "%d=\toutput: %d (%d)\tPassed ", i, n_event_align_pairs, n_event_align_pairs_out);
          // if (check_event_align_pairs(db->event_align_pairs[i], db_out->event_align_pairs[i], n_event_align_pairs) == 0)
          // {
          //   // fprintf(stderr, "%d=\t Found conflict in event_align_pairs\n", i);
          //   fprintf(stderr, "%d=\toutput: %d, expected: %d\tFailed\n", i, n_event_align_pairs, n_event_align_pairs_out);
          // }
          // else
          // {
          //   // fprintf(stderr, "%d=\t Run pass\n", i);
          //   fprintf(stderr, "%d=\toutput: %d, expected: %d\tPassed\n", i, n_event_align_pairs, n_event_align_pairs_out);
          // }
          check_event_align_pairs(db->event_align_pairs[i], db_out->event_align_pairs[i], n_event_align_pairs);
          fprintf(stderr, "\n");
        }

        // printf("readpos:%d, refpos:%d\n",db_out->event_align_pairs[0]->read_pos, db_out->event_align_pairs[0]->ref_pos);
      }
    }

    // free
    for (int i = 0; i < db->n_bam_rec; i++)
    {
      free(db->et[i].event);
    }
    free(core->model);
    free(db->read_len);
    free(db->scalings);

    free(db->f5);
    free(db->event_align_pairs);
    free(db->read);

    free(db_out->event_align_pairs);
    // free(db_out->event_align_pairs);
    free(db_out->n_event_align_pairs);
    free(db_out);

    // free(db->f5);

    free(db->et);
    // free(db->read);

    // free(db->event_align_pairs);
    free(db->n_event_align_pairs);

    free(core);
    free(db);
  }
  fprintf(stderr, "Kernel execution: %.3f seconds\n", total_kernel_time);
  // fprintf(stderr, "Total execution time for core_kernel %.3f seconds\n", align_core_kernel_time);
  // fprintf(stderr, "Total execution time for post_kernel %.3f seconds\n", align_post_kernel_time);
  // align_cl_total_kernel = align_pre_kernel_time + align_core_kernel_time + align_post_kernel_time;
  // fprintf(stderr, "Total execution time for all kernels %.3f seconds\n", align_cl_total_kernel);
  fprintf(stderr, "Data transfer: %.3f seconds\n", total_data_time);

  cleanup();
  return 0;
}

void align_ocl(core_t *core, db_t *db)
{

  // db->n_event_align_pairs[i] = align(
  //     db->event_align_pairs[i], db->read[i], db->read_len[i], db->et[i],
  //     core->model, db->scalings[i], db->f5[i]->sample_rate);

  int32_t i;
  int32_t n_bam_rec = db->n_bam_rec;
  double tick;

  tick = realtime();

  for (int read_i = 0; read_i < db->n_bam_rec; read_i++)
  {
    double data_time = 0, kernel_time = 0;
    //OUTPUT VARIABLES ##################################
    //n_event_align_pairs
    if (core->opt.verbosity > 1)
      printf("allocate: n_event_align_pairs\n");
    cl_mem n_event_align_pairs = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int32_t), NULL, &status);
    checkError(status, "Failed clCreateBuffer");

    //event_align_pairs
    if (core->opt.verbosity > 1)
      printf("allocate: event_align_pairs\n");
    cl_mem event_align_pairs = clCreateBuffer(context, CL_MEM_READ_WRITE, 2 * db->et[read_i].n * sizeof(AlignedPair), NULL, &status);
    checkError(status, "Failed clCreateBuffer");
    //###################################################

    //INPUT VARIABLES ###################################
    //read
    if (core->opt.verbosity > 1)
      printf("allocate: read\n");
    cl_mem read = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, db->read_len[read_i] * sizeof(char), db->read[read_i], &status);
    checkError(status, "Failed clCreateBuffer");

    //et-events
    if (core->opt.verbosity > 1)
      printf("allocate: events\n");
    cl_mem event = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, db->et[read_i].n * sizeof(event_t), db->et[read_i].event, &status);
    checkError(status, "Failed clCreateBuffer");

    //model
    if (core->opt.verbosity > 1)
      printf("allocate: model\n");
    cl_mem model = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, NUM_KMER * sizeof(model_t), core->model, &status);
    checkError(status, "Failed clCreateBuffer");

    //scalings
    if (core->opt.verbosity > 1)
      printf("allocate: scalings\n");
    cl_mem scalings = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(scalings_t), &db->scalings[read_i], &status);
    checkError(status, "Failed clCreateBuffer");

    size_t n_bands = db->et[read_i].n + db->read_len[read_i];

    //bands
    if (core->opt.verbosity > 1)
      printf("allocate: bands\n");
    cl_mem bands = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * n_bands * ALN_BANDWIDTH, NULL, &status);
    checkError(status, "Failed clCreateBuffer");

    //trace

    if (core->opt.verbosity > 1)
      printf("allocate: trace\n");
    uint8_t zeros[n_bands * ALN_BANDWIDTH];

    for (i = 0; i < n_bands * ALN_BANDWIDTH; i++)
      zeros[i] = 0;

    cl_mem trace = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(uint8_t) * n_bands * ALN_BANDWIDTH, zeros, &status);
    checkError(status, "Failed clCreateBuffer");

    //band_lower_left
    if (core->opt.verbosity > 1)
      printf("allocate: band_lower_left\n");
    cl_mem band_lower_left = clCreateBuffer(context, CL_MEM_READ_WRITE, n_bands * sizeof(EventKmerPair), NULL, &status);
    checkError(status, "Failed clCreateBuffer");

    //kmer_rank
    if (core->opt.verbosity > 1)
      printf("allocate: kmer_rank\n");
    cl_mem kmer_rank = clCreateBuffer(context, CL_MEM_READ_WRITE, (db->read_len[read_i] - KMER_SIZE + 1) * sizeof(size_t), NULL, &status);
    checkError(status, "Failed clCreateBuffer");

    //###############################################

    // db->n_event_align_pairs[i] = align(
    //     db->event_align_pairs[i], db->read[i], db->read_len[i], db->et[i],
    //     core->model, db->scalings[i], db->f5[i]->sample_rate);

    data_time += (realtime() - tick);

    //SET KERNEL ARGUMENTS & LAUNCH
    tick = realtime();

    if (core->opt.verbosity > 1)
      printf("kernel args: event_align_pairs\n");
    status = clSetKernelArg(align_kernel_single, 0, sizeof(cl_mem), &event_align_pairs);
    checkError(status, "Failed to set kernel args");

    if (core->opt.verbosity > 1)
      printf("kernel args: n_event_align_pairs\n");
    status = clSetKernelArg(align_kernel_single, 1, sizeof(cl_mem), &n_event_align_pairs);
    checkError(status, "Failed to set kernel args");

    if (core->opt.verbosity > 1)
      printf("kernel args: read\n");
    status = clSetKernelArg(align_kernel_single, 2, sizeof(cl_mem), &read);
    checkError(status, "Failed to set kernel args");

    if (core->opt.verbosity > 1)
      printf("kernel args: db->read_len[read_i] = %d\n", db->read_len[read_i]);
    status = clSetKernelArg(align_kernel_single, 3, sizeof(int32_t), &db->read_len[read_i]);
    checkError(status, "Failed to set kernel args");

    if (core->opt.verbosity > 1)
      printf("kernel args: event\n");
    status = clSetKernelArg(align_kernel_single, 4, sizeof(cl_mem), &event);
    checkError(status, "Failed to set kernel args");

    if (core->opt.verbosity > 1)
      printf("kernel args: db->et[read_i].n = %d\n", db->et[read_i].n);
    status = clSetKernelArg(align_kernel_single, 5, sizeof(size_t), &db->et[read_i].n);
    checkError(status, "Failed to set kernel args");

    if (core->opt.verbosity > 1)
      printf("kernel args: scalings\n");
    status = clSetKernelArg(align_kernel_single, 6, sizeof(cl_mem), &scalings);
    checkError(status, "Failed to set kernel args");

    if (core->opt.verbosity > 1)
      printf("kernel args: model\n");
    status = clSetKernelArg(align_kernel_single, 7, sizeof(cl_mem), &model);
    checkError(status, "Failed to set kernel args");

    if (core->opt.verbosity > 1)
      printf("kernel args: kmer_rank\n");
    status = clSetKernelArg(align_kernel_single, 8, sizeof(cl_mem), &kmer_rank);
    checkError(status, "Failed to set kernel args");

    if (core->opt.verbosity > 1)
      printf("kernel args: bands\n");
    status = clSetKernelArg(align_kernel_single, 9, sizeof(cl_mem), &bands);
    checkError(status, "Failed to set kernel args");

    if (core->opt.verbosity > 1)
      printf("kernel args: trace\n");
    status = clSetKernelArg(align_kernel_single, 10, sizeof(cl_mem), &trace);
    checkError(status, "Failed to set kernel args");

    if (core->opt.verbosity > 1)
      printf("kernel args: band_lower_left\n");
    status = clSetKernelArg(align_kernel_single, 11, sizeof(cl_mem), &band_lower_left);
    checkError(status, "Failed to set kernel args");

    const size_t gSize[3] = {1, 1, 1};
    const size_t wgSize[3] = {1, 1, 1};

    if (core->opt.verbosity > 1)
      printf("kernel launch\n");
    clEnqueueNDRangeKernel(queue, align_kernel_single, 1, NULL, gSize, wgSize, 0, NULL, NULL);
    status = clFinish(queue);
    checkError(status, "Failed to finish");

    kernel_time += (realtime() - tick);

    //COPY BACK

    AlignedPair *event_align_pairs_host;
    posix_memalign((void **)&event_align_pairs_host, AOCL_ALIGNMENT, 2 * db->et[read_i].n * sizeof(AlignedPair));
    MALLOC_CHK(event_align_pairs_host);
    tick = realtime();

    if (core->opt.verbosity > 1)
      printf("Copy back: n_event_align_pairs\n");
    status = clEnqueueReadBuffer(queue, n_event_align_pairs, CL_TRUE, 0, sizeof(int32_t), &db->n_event_align_pairs[read_i], 0, NULL, NULL);
    checkError(status, "clEnqueueReadBuffer");

    if (core->opt.verbosity > 1)
      printf("Copy back: event_align_pairs\n");
    status = clEnqueueReadBuffer(queue, event_align_pairs, CL_TRUE, 0, 2 * db->et[read_i].n * sizeof(AlignedPair), event_align_pairs_host, 0, NULL, NULL);
    checkError(status, "clEnqueueReadBuffer");

    data_time += (realtime() - tick);

    if (core->opt.verbosity > 1)
      printf("Release mem objects: n_event_align_pairs\n");
    status = clReleaseMemObject(n_event_align_pairs);
    checkError(status, "clReleaseMemObject failed!");

    if (core->opt.verbosity > 1)
      printf("Release mem objects: event_align_pairs\n");
    status = clReleaseMemObject(event_align_pairs);
    checkError(status, "clReleaseMemObject failed!");

    if (core->opt.verbosity > 1)
      printf("Release mem objects: read\n");
    status = clReleaseMemObject(read);
    checkError(status, "clReleaseMemObject failed!");

    if (core->opt.verbosity > 1)
      printf("Release mem objects: event\n");
    status = clReleaseMemObject(event);
    checkError(status, "clReleaseMemObject failed!");

    if (core->opt.verbosity > 1)
      printf("Release mem objects: model\n");
    status = clReleaseMemObject(model);
    checkError(status, "clReleaseMemObject failed!");

    if (core->opt.verbosity > 1)
      printf("Release mem objects: scalings\n");
    status = clReleaseMemObject(scalings);
    checkError(status, "clReleaseMemObject failed!");

    if (core->opt.verbosity > 1)
      printf("Release mem objects: bands\n");
    status = clReleaseMemObject(bands);
    checkError(status, "clReleaseMemObject failed!");

    if (core->opt.verbosity > 1)
      printf("Release mem objects: trace\n");
    status = clReleaseMemObject(trace);
    checkError(status, "clReleaseMemObject failed!");

    if (core->opt.verbosity > 1)
      printf("Release mem objects: band_lower_left\n");
    status = clReleaseMemObject(band_lower_left);
    checkError(status, "clReleaseMemObject failed!");

    if (core->opt.verbosity > 1)
      printf("Release mem objects: kmer_rank\n");
    status = clReleaseMemObject(kmer_rank);
    checkError(status, "clReleaseMemObject failed!");

    //post work
    if (core->opt.verbosity > 1)
      printf("Post work\n");
    db->event_align_pairs[read_i] = (AlignedPair *)malloc(sizeof(AlignedPair) * db->n_event_align_pairs[read_i]);
    memcpy(db->event_align_pairs[read_i], &event_align_pairs_host[read_i * 2], sizeof(AlignedPair) * db->n_event_align_pairs[read_i]);

    //free the temp arrays on host
    free(event_align_pairs_host);

    if (core->opt.verbosity > 0)
      printf("read_no: %d / %d \t read_len: %d \t time:%.3f\n", read_i, db->n_bam_rec, db->read_len[read_i], kernel_time);

    total_data_time += data_time;
    total_kernel_time += kernel_time;
  }
}

/////// HELPER FUNCTIONS ///////

bool init()
{
  cl_int status;

  if (!setCwdToExeDir())
  {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
  if (platform == NULL)
  {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL devices.
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;

  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

  // We'll just use the first device.
  device = devices[0];

  // Display some device information.
  // display_device_info(device);

  // Create the context.
  context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the command queue.
  queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  // Create the program.

  std::string binary_file = getBoardBinaryFile("align", device);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

  // Build the program that was just created.

  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  const char *kernel1_name = "align_kernel_single"; // Kernel name, as defined in the CL file

  align_kernel_single = clCreateKernel(program, kernel1_name, &status);
  checkError(status, "Failed to pre create kernel");

  return true;
}

// Free the resources allocated during initialization
void cleanup()
{

  if (align_kernel_single)
  {
    clReleaseKernel(align_kernel_single);
  }

  if (program)
  {
    clReleaseProgram(program);
  }

  if (queue)
  {
    clReleaseCommandQueue(queue);
  }
  if (context)
  {
    clReleaseContext(context);
  }
}

// Helper functions to display parameters returned by OpenCL queries
static void device_info_ulong(cl_device_id device, cl_device_info param, const char *name)
{
  cl_ulong a = 99;
  clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
  printf("%-40s = %lu\n", name, a);
}
static void device_info_size_t(cl_device_id device, cl_device_info param, const char *name)
{
  size_t a;
  clGetDeviceInfo(device, param, sizeof(size_t), &a, NULL);
  printf("%-40s = %zu\n", name, a);
}

static void device_info_size_t_arr(cl_device_id device, cl_device_info param, const char *name)
{
  size_t a[3];
  clGetDeviceInfo(device, param, sizeof(size_t) * 3, &a, NULL);
  printf("%-40s = %zu, %zu, %zu\n", name, a[0], a[1], a[2]);
}
static void device_info_uint(cl_device_id device, cl_device_info param, const char *name)
{
  cl_uint a;
  clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
  printf("%-40s = %u\n", name, a);
}
static void device_info_bool(cl_device_id device, cl_device_info param, const char *name)
{
  cl_bool a;
  clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
  printf("%-40s = %s\n", name, (a ? "true" : "false"));
}
static void device_info_string(cl_device_id device, cl_device_info param, const char *name)
{
  char a[STRING_BUFFER_LEN];
  clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &a, NULL);
  printf("%-40s = %s\n", name, a);
}

// Query and display OpenCL information on device and runtime environment
static void display_device_info(cl_device_id device)
{

  printf("Querying device for info:\n");
  printf("========================\n");
  device_info_string(device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
  device_info_string(device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
  device_info_uint(device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
  device_info_string(device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
  device_info_string(device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
  device_info_uint(device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
  device_info_bool(device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
  device_info_bool(device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
  device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
  device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
  device_info_ulong(device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
  device_info_bool(device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
  device_info_ulong(device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
  device_info_ulong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
  device_info_ulong(device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
  device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
  device_info_ulong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
  device_info_uint(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
  device_info_uint(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MEM_BASE_ADDR_ALIGN");
  device_info_uint(device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
  device_info_uint(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");
  device_info_size_t_arr(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, "CL_DEVICE_MAX_WORK_ITEM_SIZES");
  device_info_size_t(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, "CL_DEVICE_MAX_WORK_GROUP_SIZE");

  {
    cl_command_queue_properties ccp;
    clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
    printf("%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) ? "true" : "false"));
    printf("%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE) ? "true" : "false"));
  }
}
//#endif