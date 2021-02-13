#include "f5c.h"

int32_t align(AlignedPair *out_2, char *sequence, int32_t sequence_len,
              event_table events, model_t *models, scalings_t scaling,
              float sample_rate);

void load_align_arguments(core_t *core, db_t *db, int32_t i, const char *align_args_dump_dir)
{

    char foldername[100];
    snprintf(foldername, sizeof(foldername), "%s/%d", align_args_dump_dir, i);

    char filename[150];
    FILE *fp;
    size_t read_count;
    int32_t read_count2;

    // db->n_event_align_pairs[i]
    snprintf(filename, sizeof(filename), "%s/%s", foldername, "n_event_align_pairs[i].dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    read_count = fread(&(db->n_event_align_pairs[i]), sizeof(int32_t), 1, fp);
    assert(read_count == 1);
    fclose(fp);

    /// db->event_align_pairs - out_2
    snprintf(filename, sizeof(filename), "%s/%s", foldername, "event_align_pairs.dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    int32_t pairs = db->n_event_align_pairs[i];
    db->event_align_pairs[i] = (AlignedPair *)malloc(sizeof(AlignedPair) * pairs);
    read_count2 = fread(db->event_align_pairs[i], sizeof(AlignedPair), pairs, fp);
    assert(read_count2 == pairs);
    fclose(fp);

    // db->read_len[i] - sequence_len
    snprintf(filename, sizeof(filename), "%s/%s", foldername, "read_len[i].dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    read_count = fread(&(db->read_len[i]), sizeof(int32_t), 1, fp);
    assert(read_count == 1);
    fclose(fp);

    // db->read[i] - sequence
    snprintf(filename, sizeof(filename), "%s/%s", foldername, "read[i].dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    db->read[i] = (char *)malloc(sizeof(char) * (db->read_len[i] + 1));  //with null term
    read_count2 = fread(db->read[i], sizeof(char), db->read_len[i], fp); //read without null term
    db->read[i][db->read_len[i]] = '\0';
    assert(read_count2 == db->read_len[i]);
    fclose(fp);

    // db->et[i] - events
    snprintf(filename, sizeof(filename), "%s/%s", foldername, "et[i].dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    read_count = fread(&(db->et[i]), sizeof(event_table), 1, fp);
    size_t n_events = db->et[i].n;
    db->et[i].event = (event1_t *)malloc(sizeof(event1_t) * n_events);
    read_count = fread(db->et[i].event, sizeof(event1_t), n_events, fp);
    assert(read_count == n_events);
    fclose(fp);

    // db->scalings[i] - scaling
    snprintf(filename, sizeof(filename), "%s/%s", foldername, "scalings[i].dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    read_count = fread(&(db->scalings[i]), sizeof(scalings_t), 1, fp);
    assert(read_count == 1);
    fclose(fp);

    // db->f5[i]->sample_rate - sample_rate
    snprintf(filename, sizeof(filename), "%s/%s", foldername, "f5[i].sample_rate.dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    db->f5[i] = (fast5_t *)malloc(sizeof(fast5_t));
    read_count = fread(&(db->f5[i]->sample_rate), sizeof(float), 1, fp);
    assert(read_count == 1);
    fclose(fp);

    // core->model - models
    snprintf(filename, sizeof(filename), "%s/%s", align_args_dump_dir, "model.dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    read_count = fread(core->model, sizeof(model_t), NUM_KMER, fp);
    assert(read_count == NUM_KMER);
    fclose(fp);
}
/*
int32_t align(AlignedPair* out_2, char* sequence, int32_t sequence_len,
              event_table events, model_t* models, scalings_t scaling,
              float sample_rate) {*/
int check_event_align_pairs(AlignedPair *pair_1, AlignedPair *pair_2, int32_t size);

int main()
{
    const char *align_args_dump_dir = "dump_small_1";

    db_t *db;
    db = (db_t *)malloc(sizeof(db_t));
    db->n_event_align_pairs = (int32_t *)malloc(sizeof(int32_t) * 143);
    db->event_align_pairs = (AlignedPair **)malloc(sizeof(AlignedPair *) * 143);
    db->read_len = (int32_t *)malloc(sizeof(AlignedPair) * 143);
    db->read = (char **)malloc(sizeof(char *) * 143);
    db->et = (event_table *)malloc(sizeof(event_table) * 143);
    db->scalings = (scalings_t *)malloc(sizeof(scalings_t) * 143);
    db->f5 = (fast5_t **)malloc(sizeof(fast5_t *) * 143);

    core_t *core;
    core = (core_t *)malloc(sizeof(core_t));
    core->model = (model_t *)malloc(sizeof(model_t) * 143);

    db_t *db_out;
    db_out = (db_t *)malloc(sizeof(db_t));
    db_out->n_event_align_pairs = (int32_t *)malloc(sizeof(int32_t) * 143);
    db_out->event_align_pairs = (AlignedPair **)malloc(sizeof(AlignedPair *) * 143);

    //create necessary inputs

    for (int i = 0; i < 143; i++)
    {

        load_align_arguments(core, db, i, align_args_dump_dir);

        int32_t pairs = db->n_event_align_pairs[i];

        db_out->event_align_pairs[i] = (AlignedPair *)malloc(sizeof(AlignedPair) * pairs);

        //call align function and store the output
        db_out->n_event_align_pairs[i] =
            align(db_out->event_align_pairs[i], db->read[i], db->read_len[i], db->et[i],
                  core->model, db->scalings[i], db->f5[i]->sample_rate);

        //compare with original output

        int32_t n_event_align_pairs = db->n_event_align_pairs[i];
        int32_t n_event_align_pairs_out = db_out->n_event_align_pairs[i];

        if (n_event_align_pairs != n_event_align_pairs_out)
        {
            fprintf(stderr, "%d = %d (%d)\t Failed\n", i, n_event_align_pairs_out, n_event_align_pairs);
            // break;
        }
        else
        {
            fprintf(stderr, "%d = %d (%d)\t Passed\n", i, n_event_align_pairs_out, n_event_align_pairs);
            // if (check_event_align_pairs(db_out->event_align_pairs[i], db->event_align_pairs[i], pairs) == 0)
            // {
            //     fprintf(stderr, "%d (%d)\t Failed\n", i);
            // }
            // else
            // {
            //     fprintf(stderr, "%dPassed\n", i);
            // }
        }
    }
}

int check_event_align_pairs(AlignedPair *pair_1, AlignedPair *pair_2, int32_t size)
{
    int i;
    for (i = 0; i < size; i++)
    {
        if (pair_1[i].read_pos != pair_2[i].read_pos || pair_1[i].ref_pos != pair_2[i].ref_pos)
        {
            return 0;
        }
    }
    return 1;
}
