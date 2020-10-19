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

// #include "dump_functions.h"

using namespace aocl_utils;

#include "error.h"
// #include "f5c.h"
// #include "f5cmisc.cuh"
#include "f5cmisc.h"


#ifndef CPU_GPU_PROC

#define STRING_BUFFER_LEN 1024

// Runtime constants
// Used to define the work set over which this kernel will execute.
static const size_t work_group_size = 8;  // 8 threads in the demo workgroup
// Defines kernel argument value, which is the workitem ID that will
// execute a printf call
static const int thread_id_to_output = 2;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_kernel kernel = NULL;
static cl_program program = NULL;

// Function prototypes
bool init();
void cleanup();
static void device_info_ulong( cl_device_id device, cl_device_info param, const char* name);
static void device_info_uint( cl_device_id device, cl_device_info param, const char* name);
static void device_info_bool( cl_device_id device, cl_device_info param, const char* name);
static void device_info_string( cl_device_id device, cl_device_info param, const char* name);
static void display_device_info( cl_device_id device );

static void align_cuda(core_t *core, db_t *db);

// Entry point.
int main() {
  cl_int status;

  const char * align_args_dump_dir = "align_args_dump";

  if(!init()) {
    return -1;
  }

  core_t * core;
  db_t * db;
  align_cuda(core, db);

  return 0;
}

void align_cuda(core_t *core, db_t *db) {
    
    int32_t i;
    int32_t n_bam_rec = db->n_bam_rec;
    double realtime1;

    /**cuda pointers*/
    char* host_read;        //flattened reads sequences
    ptr_t* host_read_ptr; //index pointer for flattedned "reads"
    int32_t* host_read_len;
    int64_t sum_read_len;
    int32_t* host_n_events;
    event_t* host_event_table;
    ptr_t* host_event_ptr;
    int64_t sum_n_events;
    scalings_t* host_scalings;
    AlignedPair* host_event_align_pairs;
    int32_t* host_n_event_align_pairs;
    float * host_bands;
    uint8_t * host_trace;
    EventKmerPair* host_band_lower_left;

   // realtime1 = realtime();

    //int32_t cuda_device_num = core->opt.cuda_dev_id;

    // |||||||||||||||||OpenCL Initialization||||||||||||||||||||||||||||
    

    // ||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||

    ptr_t* read_ptr_host = (ptr_t*)malloc(sizeof(ptr_t) * n_bam_rec);
    // MALLOC_CHK(read_ptr_host);
// #endif
    sum_read_len = 0;

    //read sequences : needflattening
    for (i = 0; i < n_bam_rec; i++) {
        read_ptr_host[i] = sum_read_len;
        sum_read_len += (db->read_len[i] + 1); //with null term
    }
    //form the temporary flattened array on host
    char* read_host = (char*)malloc(sizeof(char) * sum_read_len);
    MALLOC_CHK(read_host);
    for (i = 0; i < n_bam_rec; i++) {
        ptr_t idx = read_ptr_host[i];
        strcpy(&read_host[idx], db->read[i]);
    }

        //now the events : need flattening
    //num events : need flattening
    //get the total size and create the pointers
// #ifdef CUDA_PRE_MALLOC
//     int32_t* n_events_host = core->cuda->n_events_host;
//     ptr_t* event_ptr_host = core->cuda->event_ptr_host;
// #else
    int32_t* n_events_host = (int32_t*)malloc(sizeof(int32_t) * n_bam_rec);
    MALLOC_CHK(n_events_host);
    ptr_t* event_ptr_host = (ptr_t*)malloc(sizeof(ptr_t) * n_bam_rec);
    MALLOC_CHK(event_ptr_host);
// #endif

   sum_n_events = 0;
    for (i = 0; i < n_bam_rec; i++) {
        n_events_host[i] = db->et[i].n;
        event_ptr_host[i] = sum_n_events;
        sum_n_events += db->et[i].n;
    }

    //event table flatten
    //form the temporary flattened array on host
    event_t* event_table_host =(event_t*)malloc(sizeof(event_t) * sum_n_events);
    MALLOC_CHK(event_table_host);
    for (i = 0; i < n_bam_rec; i++) {
        ptr_t idx = event_ptr_host[i];
        memcpy(&event_table_host[idx], db->et[i].event,
               sizeof(event_t) * db->et[i].n);
    }

    AlignedPair* event_align_pairs_host =
        (AlignedPair*)malloc(2 * sum_n_events * sizeof(AlignedPair));
    MALLOC_CHK(event_align_pairs_host);

  //core->align_cuda_preprocess += (realtime() - realtime1);

    /** Start GPU mallocs**/
  //realtime1 = realtime();
/*
cudaError_t cudaMalloc 	(void ** devPtr,size_t size)
*/


    if(core->opt.verbosity>1) print_size("read_ptr array",n_bam_rec * sizeof(ptr_t));
    cl_mem read_ptr = clCreateBuffer(context, CL_MEM_READ_WRITE, n_bam_rec * sizeof(ptr_t), host_read_ptr, NULL);
    //CUDA_CHK();

    if(core->opt.verbosity>1) print_size("read_lens",n_bam_rec * sizeof(int32_t));
    cl_mem read_len = clCreateBuffer(context, CL_MEM_READ_WRITE, n_bam_rec * sizeof(int32_t), host_read_len, NULL);
    //CUDA_CHK();
    //n_events
    if(core->opt.verbosity>1) print_size("n_events",n_bam_rec * sizeof(int32_t));
    cl_mem n_events = clCreateBuffer(context, CL_MEM_READ_WRITE, n_bam_rec * sizeof(int32_t), host_n_events, NULL);
    //CUDA_CHK();
    //event ptr
    if(core->opt.verbosity>1) print_size("event ptr",n_bam_rec * sizeof(ptr_t));
    cl_mem event_ptr = clCreateBuffer(context, CL_MEM_READ_WRITE, n_bam_rec * sizeof(ptr_t), host_event_ptr, NULL);
   // CUDA_CHK();
    //scalings : already linear
    if(core->opt.verbosity>1) print_size("Scalings",n_bam_rec * sizeof(scalings_t));
    cl_mem scalings = clCreateBuffer(context, CL_MEM_READ_WRITE, n_bam_rec * sizeof(scalings_t), host_scalings, NULL);
    //CUDA_CHK();
    //model : already linear
    model_t* host_model;
    cl_mem model = clCreateBuffer(context, CL_MEM_READ_WRITE, NUM_KMER * sizeof(model_t), host_model, NULL);
    //CUDA_CHK();

    if(core->opt.verbosity>1) print_size("read array",sum_read_len * sizeof(char));
    // cudaMalloc((void**)&read, sum_read_len * sizeof(char)); //with null char
    // CUDA_CHK();
    cl_mem read = clCreateBuffer(context, CL_MEM_READ_WRITE, sum_read_len * sizeof(char), host_read, NULL);
    if(core->opt.verbosity>1) print_size("event table",sum_n_events * sizeof(event_t));
    // cudaMalloc((void**)&event_table, sum_n_events * sizeof(event_t));
    // CUDA_CHK();
    cl_mem event_table = clCreateBuffer(context, CL_MEM_READ_WRITE, sum_n_events * sizeof(event_t), host_event_table, NULL);
    model_t* host_model_kmer_cache;
    // cudaMalloc((void**)&model_kmer_cache, sum_read_len * sizeof(model_t));
    // CUDA_CHK();
    cl_mem model_kmer_cache = clCreateBuffer(context, CL_MEM_READ_WRITE, sum_read_len * sizeof(model_t), host_model_kmer_cache, NULL);

    /**allocate output arrays for cuda**/
    if(core->opt.verbosity>1) print_size("event align pairs",2 * sum_n_events *sizeof(AlignedPair));
    // cudaMalloc((void**)&event_align_pairs,2 * sum_n_events *sizeof(AlignedPair)); //todo : need better huristic
    // CUDA_CHK();
    cl_mem event_align_pairs = clCreateBuffer(context, CL_MEM_READ_WRITE, sum_n_events * sizeof(AlignedPair), host_event_align_pairs, NULL);
// #ifdef CUDA_PRE_MALLOC
//     n_event_align_pairs=core->cuda->n_event_align_pairs;
// #else
    if(core->opt.verbosity>1) print_size("n_event_align_pairs",n_bam_rec * sizeof(int32_t));
    // cudaMalloc((void**)&n_event_align_pairs, n_bam_rec * sizeof(int32_t));
    // CUDA_CHK();
    cl_mem n_event_align_pairs=clCreateBuffer(context, CL_MEM_READ_WRITE, n_bam_rec * sizeof(int32_t), host_n_event_align_pairs, NULL);
// #endif
    //scratch arrays
    size_t sum_n_bands = sum_n_events + sum_read_len; //todo : can be optimised
    if(core->opt.verbosity>1) print_size("bands",sizeof(float) * sum_n_bands * ALN_BANDWIDTH);
    // cudaMalloc((void**)&bands,sizeof(float) * sum_n_bands * ALN_BANDWIDTH);
    // CUDA_CHK();
    cl_mem bands=clCreateBuffer(context, CL_MEM_READ_WRITE, sum_n_bands * sizeof(float), host_bands, NULL);
    if(core->opt.verbosity>1) print_size("trace",sizeof(uint8_t) * sum_n_bands * ALN_BANDWIDTH);
    // cudaMalloc((void**)&trace, sizeof(uint8_t) * sum_n_bands * ALN_BANDWIDTH);
    // CUDA_CHK();
    cl_mem trace =clCreateBuffer(context, CL_MEM_READ_WRITE, sum_n_bands * sizeof(uint8_t), host_trace, NULL);



    // cudaMemset(trace,0,sizeof(uint8_t) * sum_n_bands * ALN_BANDWIDTH); //initialise the trace array to 0
    
    size_t trace_size = sizeof(uint8_t) * sum_n_bands * ALN_BANDWIDTH;
    cl_mem trace_buffer= clCreateBuffer(context, CL_MEM_READ_WRITE, trace_size, trace, NULL);
    uint8_t zero = 0;
    clEnqueueFillBuffer(queue, trace_buffer, &zero, trace_size, 0, trace_size, NULL, NULL, NULL);



    if(core->opt.verbosity>1) print_size("band_lower_left",sizeof(EventKmerPair)* sum_n_bands);
    // cudaMalloc((void**)&band_lower_left, sizeof(EventKmerPair)* sum_n_bands);
    // CUDA_CHK();
    cl_mem band_lower_left=clCreateBuffer(context, CL_MEM_READ_WRITE, sum_n_bands * sizeof(EventKmerPair), host_band_lower_left, NULL);





}


/////// HELPER FUNCTIONS ///////

bool init() {
  cl_int status;

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA SDK for OpenCL(TM)");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }

  // User-visible output - Platform information
  {
    char char_buffer[STRING_BUFFER_LEN]; 
    printf("Querying platform for info:\n");
    printf("==========================\n");
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
    printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);
  }

  // Query the available OpenCL devices.
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;

  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

  // We'll just use the first device.
  device = devices[0];

  // Display some device information.
  display_device_info(device);

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
  const char *kernel_name = "align";  // Kernel name, as defined in the CL file
  kernel = clCreateKernel(program, kernel_name, &status);
  checkError(status, "Failed to create kernel");

  return true;
}

// Free the resources allocated during initialization
void cleanup() {
  if(kernel) {
    clReleaseKernel(kernel);  
  }
  if(program) {
    clReleaseProgram(program);
  }
  if(queue) {
    clReleaseCommandQueue(queue);
  }
  if(context) {
    clReleaseContext(context);
  }
}

// Helper functions to display parameters returned by OpenCL queries
static void device_info_ulong( cl_device_id device, cl_device_info param, const char* name) {
   cl_ulong a;
   clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
   printf("%-40s = %lu\n", name, a);
}
static void device_info_uint( cl_device_id device, cl_device_info param, const char* name) {
   cl_uint a;
   clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
   printf("%-40s = %u\n", name, a);
}
static void device_info_bool( cl_device_id device, cl_device_info param, const char* name) {
   cl_bool a;
   clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
   printf("%-40s = %s\n", name, (a?"true":"false"));
}
static void device_info_string( cl_device_id device, cl_device_info param, const char* name) {
   char a[STRING_BUFFER_LEN]; 
   clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &a, NULL);
   printf("%-40s = %s\n", name, a);
}

// Query and display OpenCL information on device and runtime environment
static void display_device_info( cl_device_id device ) {

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

   {
      cl_command_queue_properties ccp;
      clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
      printf("%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)?"true":"false"));
      printf("%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE)?"true":"false"));
   }
}

#endif