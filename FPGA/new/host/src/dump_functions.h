

void load_align_arguments(core_t *core, db_t *db, int32_t i, const char * align_args_dump_dir) {

    char foldername[40];
    snprintf(foldername, sizeof(foldername), "%s/%d", align_args_dump_dir, i);

    char filename[50];
    FILE * fp;
    size_t read_count;
    int32_t read_count2;

    // db->read_len[i] - sequence_len
    snprintf(filename, sizeof(filename), "%s/%s", foldername, "read_len[i].dat");
    fp = fopen(filename, "r");
    if(fp==NULL)printf("Can not open %s\n", filename);
    read_count = fread(&(db->read_len[i]), sizeof(int32_t), 1, fp);
    assert(read_count == 1);
    fclose(fp);
    
    // db->read[i] - sequence
    snprintf(filename, sizeof(filename), "%s/%s", foldername, "read[i].dat");
    fp = fopen(filename, "r");
    if(fp==NULL)printf("Can not open %s\n", filename);
    int32_t read_len = db->read_len[i];
    db->read[i] = (char*)malloc(sizeof(char)*read_len);
    read_count2 = fread(db->read[i], sizeof(char), read_len, fp);
    assert(read_count2 == read_len);
    fclose(fp);
    
    // db->et[i] - events
    snprintf(filename, sizeof(filename), "%s/%s", foldername, "et[i].dat");
    fp = fopen(filename, "r");
    if(fp==NULL)printf("Can not open %s\n", filename);
    read_count = fread(&(db->et[i]), sizeof(event_table), 1, fp);
    size_t n_events = db->et[i].n;
    db->et[i].event = (event_t*)malloc(sizeof(event_t)*n_events);
    read_count = fread(db->et[i].event, sizeof(event_t), n_events, fp);
    assert(read_count == n_events);
    fclose(fp);
    
    // core->model - models
    snprintf(filename, sizeof(filename), "%s/%s", foldername, "model.dat");
    fp = fopen(filename, "r");
    if(fp==NULL)printf("Can not open %s\n", filename);
    core->model = (model_t*)malloc(sizeof(model_t)*NUM_KMER);
    read_count = fread(core->model, sizeof(model_t), NUM_KMER, fp);
    assert(read_count == NUM_KMER);
    fclose(fp);

    // db->scalings[i] - scaling
    snprintf(filename, sizeof(filename), "%s/%s", foldername, "scalings[i].dat");
    fp = fopen(filename, "r");
    if(fp==NULL)printf("Can not open %s\n", filename);
    read_count = fread(&(db->scalings[i]), sizeof(scalings_t), 1, fp);
    assert(read_count == 1);
    fclose(fp);

    // db->f5[i]->sample_rate - sample_rate
    snprintf(filename, sizeof(filename), "%s/%s", foldername, "f5[i].sample_rate.dat");
    fp = fopen(filename, "r");
    if(fp==NULL)printf("Can not open %s\n", filename);
    db->f5[i] = (fast5_t*)malloc(sizeof(fast5_t));
    read_count = fread(&(db->f5[i]->sample_rate), sizeof(float), 1, fp);
    assert(read_count == 1);
    fclose(fp); 
}

void load_align_outputs(db_t *db, int32_t i, const char * align_args_dump_dir) {

    char foldername[40];
    snprintf(foldername, sizeof(foldername), "%s/%d", align_args_dump_dir, i);

    char filename[50];
    FILE * fp;
    size_t read_count;
    int32_t read_count2;

    // db->n_event_align_pairs[i]
    snprintf(filename, sizeof(filename), "%s/%s", foldername, "n_event_align_pairs[i].dat");
    fp = fopen(filename, "r");
    if(fp==NULL)printf("Can not open %s\n", filename);
    read_count = fread(&(db->n_event_align_pairs[i]), sizeof(int32_t), 1, fp);
    assert(read_count == 1);
    fclose(fp);

    // db->event_align_pairs - out_2
    snprintf(filename, sizeof(filename), "%s/%s", foldername, "event_align_pairs.dat");
    fp = fopen(filename, "r");
    if(fp==NULL)printf("Can not open %s\n", filename);
    int32_t pairs = db->n_event_align_pairs[i];
    db->event_align_pairs[i] = (AlignedPair *)malloc(sizeof(AlignedPair)*pairs);
    read_count2 = fread(db->event_align_pairs[i], sizeof(AlignedPair), pairs, fp);
    assert(read_count2 == pairs);
    fclose(fp);
}


int check_event_align_pairs(AlignedPair* pair_1,AlignedPair* pair_2,int32_t size){
    int i;
    for (i=0;i<size;i++){
        if(pair_1[i].read_pos!=pair_2[i].read_pos || pair_1[i].ref_pos!=pair_2[i].ref_pos){
            return 0;
            }
        }
    return 1;
}