#pragma once

#include "config.h"

#include <complex.h>

typedef struct
{
    float freq;
    float phase;
    float phases[BLKSZ];
} coastas_t;

typedef struct
{
    struct input_t *input;
    float complex buffer[FFT_FM][BLKSZ];
    unsigned int idx;
    int psmi;
    int pli;
    int hppi;
    int aabi;
    int rdbi;
    int cfo_wait;
    unsigned int bc;
    unsigned int offset_history;
    int samperr;
    float angle;

    float alpha;
    float beta;
    coastas_t loop[FFT_FM];

    int mer_cnt;
    int mer_known;
    float error_lb;
    float error_ub;
} sync_t;

void sync_adjust(sync_t *st, int sample_adj);
void sync_push(sync_t *st, float complex *fft);
void sync_reset(sync_t *st);
void sync_init(sync_t *st, struct input_t *input);
