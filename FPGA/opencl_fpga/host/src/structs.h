
#define NUM_KMER 4096   //num k-mers for 6-mers DNA
#define CACHED_LOG 1 //if the log values of scalings and the model k-mers are cached

const int ARRAY_SIZE = 10;

struct twoArrays
{
    cl_float a[ARRAY_SIZE];
    cl_float b[ARRAY_SIZE];
    cl_float result[ARRAY_SIZE];
};

//from nanopolish
typedef struct
{
    cl_int ref_pos;
    cl_int read_pos;
} AlignedPair;

// a single event : adapted from taken from scrappie
typedef struct
{
    uint64_t start;
    cl_float length; //todo : cant be made int?
    cl_float mean;
    cl_float stdv;
    //int32_t pos;   //todo : always -1 can be removed
    //int32_t state; //todo : always -1 can be removed
} event_t;

// event table : adapted from scrappie
typedef struct
{
    size_t n;     //todo : int32_t not enough?
    size_t start; //todo : always 0?
    size_t end;   //todo : always equal to n?
    event_t *event;
} event_table;

//k-mer model
typedef struct
{
    cl_float level_mean;
    cl_float level_stdv;

#ifdef CACHED_LOG
    //calculated for efficiency
    cl_float level_log_stdv;
#endif

#ifdef LOAD_SD_MEANSSTDV
    //float sd_mean;
    //float sd_stdv;
    //float weight;
#endif
} model_t;

//scaling parameters for the signal : taken from nanopolish
typedef struct
{
    // direct parameters that must be set
    cl_float scale;
    cl_float shift;
    //float drift; = 0 always?
    cl_float var; // set later when calibrating
    //float scale_sd;
    //float var_sd;

    // derived parameters that are cached for efficiency
#ifdef CACHED_LOG
    cl_float log_var;
#endif
    //float scaled_var;
    //float log_scaled_var;
} scalings_t;


typedef struct {

    float sample_rate;

} fast5_t;

// a data batch (dynamic data based on the reads)
typedef struct {

    //read sequence //todo : optimise by grabbing it from bam seq. is it possible due to clipping?
    char** read;
    int32_t* read_len;
    int64_t* read_idx; //the index of the read entry in the BAM file

    // fast5 file //should flatten this to reduce mallocs
    fast5_t** f5;

    //event table
    event_table* et;

    //scaling
    scalings_t* scalings;

    //aligned pairs
    AlignedPair** event_align_pairs;
    int32_t* n_event_align_pairs;

    int32_t* n_event_alignment;


} db_t;

//core data structure (mostly static data throughout the program lifetime)
typedef struct {
   
    // models
    model_t* model; //dna model
;

} core_t;