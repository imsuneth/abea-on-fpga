void load_n_bam_rec(db_t *db, const char *align_args_dump_dir)
{
    char foldername[40];
    snprintf(foldername, sizeof(foldername), "%s", align_args_dump_dir);

    char filename[50];
    FILE *fp;
    size_t read_count;
    int32_t read_count2;

    // db->n_bam_rec
    snprintf(filename, sizeof(filename), "%s/%s", foldername, "n_bam_rec.dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    read_count = fread(&(db->n_bam_rec), sizeof(int32_t), 1, fp);
    assert(read_count == 1);
    fclose(fp);
}

void load_core(core_t *core, const char *align_args_dump_dir)
{
    char foldername[40];
    snprintf(foldername, sizeof(foldername), "%s", align_args_dump_dir);

    char filename[50];
    FILE *fp;
    size_t read_count;
    int32_t read_count2;

    // core->model - models
    snprintf(filename, sizeof(filename), "%s/%s", foldername, "model.dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    read_count = fread(core->model, sizeof(model_t), NUM_KMER, fp);
    assert(read_count == NUM_KMER);
    fclose(fp);
}

void load_align_arguments(db_t *db, int32_t i, const char *align_args_dump_dir)
{

    char foldername[40];
    snprintf(foldername, sizeof(foldername), "%s/%d", align_args_dump_dir, i);

    char filename[50];
    FILE *fp;
    size_t read_count;
    int32_t read_count2;

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
    db->et[i].event = (event_t *)malloc(sizeof(event_t) * n_events);
    read_count = fread(db->et[i].event, sizeof(event_t), n_events, fp);
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
}

void load_align_outputs(db_t *db, int32_t i, const char *align_args_dump_dir)
{

    char foldername[40];
    snprintf(foldername, sizeof(foldername), "%s/%d", align_args_dump_dir, i);

    char filename[50];
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

    // db->event_align_pairs - out_2
    snprintf(filename, sizeof(filename), "%s/%s", foldername, "event_align_pairs.dat");
    fp = fopen(filename, "r");
    if (fp == NULL)
        printf("Can not open %s\n", filename);
    int32_t pairs = db->n_event_align_pairs[i];
    db->event_align_pairs[i] = (AlignedPair *)malloc(sizeof(AlignedPair) * pairs);
    read_count2 = fread(db->event_align_pairs[i], sizeof(AlignedPair), pairs, fp);
    assert(read_count2 == pairs);
    fclose(fp);
}

int check_event_align_pairs(AlignedPair *pair_1, AlignedPair *pair_2, int32_t size)
{
    int i;
    int passed = size;
    for (i = 0; i < size; i++)
    {

        if (pair_1[i].read_pos != pair_2[i].read_pos || pair_1[i].ref_pos != pair_2[i].ref_pos)
        {
            // fprintf(stderr, "read_pos:%d (%d), ref_pos:%d (%d)\tFailed\n", pair_1[i].read_pos, pair_2[i].read_pos, pair_1[i].ref_pos, pair_2[i].ref_pos);
            passed--;
            // return 0;
        }
        // fprintf(stderr, "read_pos:%d (%d), ref_pos:%d (%d)\tPasses\n", pair_1[i].read_pos, pair_2[i].read_pos, pair_1[i].ref_pos, pair_2[i].ref_pos);
    }
    fprintf(stderr, "%d%\n", passed * 100 / size);
    // return 1;
}