/*
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <math.h>
#include <string.h>

#include "acquire.h"
#include "defines.h"
#include "input.h"
#include "private.h"

#define FILTER_DELAY 15
#define DECIMATION_FACTOR_FM 2
#define DECIMATION_FACTOR_AM 32

// start: 10000hz end: 163890hz
static float complex filter_lower_taps_fm[] = {
    (0.0001491737348260358-0.00023344348301179707j),
    (0.0006922566099092364+7.773741526762024e-05j),
    (0.00032526138238608837+0.0009051971137523651j),
    (-0.0003446308837737888+0.00035608699545264244j),
    (0.001513275085017085+0.0004885241505689919j),
    (0.0009233292075805366+0.006344289518892765j),
    (-0.01246884185820818+0.008554373867809772j),
    (-0.02495468407869339-0.014047245495021343j),
    (0.002645110944285989-0.04725700244307518j),
    (0.06516436487436295-0.027618050575256348j),
    (0.07382321357727051+0.0638570562005043j),
    (-0.031845834106206894+0.12149399518966675j),
    (-0.14885567128658295+0.030241960659623146j),
    (-0.10598291456699371-0.137343630194664j),
    (0.08277542144060135-0.16842930018901825j),
    (0.19261953234672546+3.2146851935976883e-07j),
    (0.08277485519647598+0.16842956840991974j),
    (-0.10598336905241013+0.13734328746795654j),
    (-0.14885558187961578-0.03024245798587799j),
    (-0.03184542432427406-0.1214940994977951j),
    (0.07382342964410782-0.0638568103313446j),
    (0.06516426801681519+0.02761826664209366j),
    (0.0026449530851095915+0.047257013618946075j),
    (-0.024954723194241524+0.014047173783183098j),
    (-0.01246881764382124-0.008554410189390182j),
    (0.0009233473101630807-0.006344286724925041j),
    (0.0015132765984162688-0.0004885198432020843j),
    (-0.0003446298942435533-0.00035608798498287797j),
    (0.000325263972626999-0.0009051961824297905j),
    (0.0006922568427398801-7.773543620714918e-05j),
    (0.00014917329826857895+0.00023344377405010164j),
    (0.0+0.0j)
};

// start: -163890hz end: -10000hz
static float complex filter_upper_taps_fm[] = {
    (0.0001491737348260358+0.00023344348301179707j),(0.0006922566099092364-7.773741526762024e-05j),(0.00032526138238608837-0.0009051971137523651j),(-0.0003446308837737888-0.00035608699545264244j),(0.001513275085017085-0.0004885241505689919j),(0.0009233292075805366-0.006344289518892765j),(-0.01246884185820818-0.008554373867809772j),(-0.02495468407869339+0.014047245495021343j),(0.002645110944285989+0.04725700244307518j),(0.06516436487436295+0.027618050575256348j),(0.07382321357727051-0.0638570562005043j),(-0.031845834106206894-0.12149399518966675j),(-0.14885567128658295-0.030241960659623146j),(-0.10598291456699371+0.137343630194664j),(0.08277542144060135+0.16842930018901825j),(0.19261953234672546-3.2146851935976883e-07j),(0.08277485519647598-0.16842956840991974j),(-0.10598336905241013-0.13734328746795654j),(-0.14885558187961578+0.03024245798587799j),(-0.03184542432427406+0.1214940994977951j),(0.07382342964410782+0.0638568103313446j),(0.06516426801681519-0.02761826664209366j),(0.0026449530851095915-0.047257013618946075j),(-0.024954723194241524-0.014047173783183098j),(-0.01246881764382124+0.008554410189390182j),(0.0009233473101630807+0.006344286724925041j),(0.0015132765984162688+0.0004885198432020843j),(-0.0003446298942435533+0.00035608798498287797j),(0.000325263972626999+0.0009051961824297905j),(0.0006922568427398801+7.773543620714918e-05j),(0.00014917329826857895-0.00023344377405010164j),(0.0+0.0j)
};


static float filter_taps_am[] = {
    -0.00038464731187559664,
    -0.00021618751634377986,
    0.0026779419276863337,
    -0.00029802651260979474,
    -0.0012626448879018426,
    -0.0013182522961869836,
    -0.012252614833414555,
    0.015980124473571777,
    0.037112727761268616,
    -0.05451361835002899,
    -0.05804193392395973,
    0.11320608854293823,
    0.055298302322626114,
    -0.16878043115139008,
    -0.022917453199625015,
    0.19178225100040436,
    -0.022917453199625015,
    -0.16878043115139008,
    0.055298302322626114,
    0.11320608854293823,
    -0.05804193392395973,
    -0.05451361835002899,
    0.037112727761268616,
    0.015980124473571777,
    -0.012252614833414555,
    -0.0013182522961869836,
    -0.0012626448879018426,
    -0.00029802651260979474,
    0.0026779419276863337,
    -0.00021618751634377986,
    -0.00038464731187559664,
    0
};

void peak_mag(acquire_t *st, float* max_mag, float complex* max_v, int *samperr)
{
    int i, j;

    memset(st->sums, 0, sizeof(float complex) * st->fftcp);
    for (i = 0; i < st->fftcp; ++i)
    {
        for (j = 0; j < ACQUIRE_SYMBOLS; ++j)
            st->sums[i] += st->buffer[i + j * st->fftcp] * conjf(st->buffer[i + j * st->fftcp + st->fft]);
    }

    for (i = 0; i < st->fftcp; ++i)
    {
        float mag;
        float complex v = 0;

        for (j = 0; j < st->cp; ++j)
            v += st->sums[(i + j) % st->fftcp] * st->shape[j] * st->shape[j + st->fft];

        mag = normf(v);
        if (mag > *max_mag)
        {
            *max_mag = mag;
            *max_v = v;
            *samperr = (i + st->fftcp - FILTER_DELAY) % st->fftcp;
        }
    }
}

void acquire_process(acquire_t *st)
{
    float complex max_v = 0, phase_increment;
    float angle, angle_diff, angle_factor;
    int samperr = 0;
    int i, j, keep;

    if (st->idx != (unsigned int)st->fftcp * (ACQUIRE_SYMBOLS + 1))
        return;

    output_advance(st->input->output);

    if (st->input->sync_state == SYNC_STATE_FINE)
    {
        samperr = st->fftcp / 2 + st->input->sync.samperr;
        st->input->sync.samperr = 0;

        angle_diff = -st->input->sync.angle;
        st->input->sync.angle = 0;
        angle = st->prev_angle + angle_diff;
        st->prev_angle = angle;
    }
    else
    {
        float complex max_v_lb = 0, max_v_ub = 0;
        float max_mag_lb = -1.0f, max_mag_ub = -1.0f;
        int samperr_lb, samperr_ub = 0;

        float complex y;
        for (i = 0; i < st->fftcp * (ACQUIRE_SYMBOLS + 1); i++)
        {
            fir_cf32_c_execute((st->mode == NRSC5_MODE_FM) ? st->filter_fm_lower : st->filter_am, &st->in_buffer[i], &y);
            st->buffer[i] = (st->mode == NRSC5_MODE_FM) ? conjf(y) : y;
        }

        peak_mag(st, &max_mag_lb, &max_v_lb, &samperr_lb);
        printf("lower: max_mag: %0.4f samperr: %d\n", max_mag_lb, samperr_lb);

        for (i = 0; i < st->fftcp * (ACQUIRE_SYMBOLS + 1); i++)
        {
            fir_cf32_c_execute((st->mode == NRSC5_MODE_FM) ? st->filter_fm_upper : st->filter_am, &st->in_buffer[i], &y);
            st->buffer[i] = (st->mode == NRSC5_MODE_FM) ? conjf(y) : y;
        }

        peak_mag(st, &max_mag_ub, &max_v_ub, &samperr_ub);
        printf("upper: max_mag: %0.4f samperr: %d\n", max_mag_ub, samperr_ub);

        if (max_mag_ub > max_mag_lb)
        {
            max_v = max_v_ub;
            samperr = samperr_ub;
        }
        else
        {
            max_v = max_v_lb;
            samperr = samperr_lb;
        }

        angle_diff = cargf(max_v * cexpf(I * -st->prev_angle));
        angle_factor = (st->prev_angle) ? 0.25 : 1.0;
        angle = st->prev_angle + (angle_diff * angle_factor);
        st->prev_angle = angle;
        input_set_sync_state(st->input, SYNC_STATE_COARSE);
    }

    for (i = 0; i < st->fftcp * (ACQUIRE_SYMBOLS + 1); i++)
        st->buffer[i] = (st->mode == NRSC5_MODE_FM) ? conjf(st->in_buffer[i]) : st->in_buffer[i];

    sync_adjust(&st->input->sync, st->fftcp / 2 - samperr);
    angle -= 2 * M_PI * st->cfo;

    st->phase *= cexpf(-(st->fftcp / 2 - samperr) * angle / st->fft * I);

    phase_increment = cexpf(angle / st->fft * I);

    if (st->mode == NRSC5_MODE_AM)
    {
        float y, sum_y = 0, sum_xy = 0, sum_x2 = 0;
        float complex last_carrier;
        float complex temp_phase = st->phase;
        float mag_sums[FFT_AM] = {0};

        for (i = 0; i < ACQUIRE_SYMBOLS; ++i)
        {
            int offset = (st->mode == NRSC5_MODE_FM) ? 0 : (FFT_AM - CP_AM) / 2;
            for (j = 0; j < st->fftcp; ++j)
            {
                float complex sample = temp_phase * st->buffer[i * st->fftcp + j + samperr];
                if (j < st->cp)
                    st->fftin[(j + offset) % st->fft] = st->shape[j] * sample;
                else if (j < st->fft)
                    st->fftin[(j + offset) % st->fft] = sample;
                else
                    st->fftin[(j + offset) % st->fft] += st->shape[j] * sample;

                temp_phase *= phase_increment;
            }
            temp_phase /= cabsf(temp_phase);

            fftwf_execute((st->mode == NRSC5_MODE_FM) ? st->fft_plan_fm : st->fft_plan_am);
            fftshift(st->fftout, st->fft);

            float x = st->fftcp * (i - (float) (ACQUIRE_SYMBOLS - 1) / 2);
            if (i == 0)
                y = cargf(st->fftout[CENTER_AM]);
            else
                y += cargf(st->fftout[CENTER_AM] / last_carrier);
            last_carrier = st->fftout[CENTER_AM];

            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;

            if (st->input->sync_state != SYNC_STATE_FINE)
            {
                for (j = CENTER_AM - PIDS_OUTER_INDEX_AM; j <= CENTER_AM + PIDS_OUTER_INDEX_AM; j++)
                {
                    mag_sums[j] += cabsf(st->fftout[j]);
                }
            }
        }

        if (st->input->sync_state != SYNC_STATE_FINE)
        {
            float max_mag = -1.0f;
            int max_index = -1;
            for (j = CENTER_AM - PIDS_OUTER_INDEX_AM; j <= CENTER_AM + PIDS_OUTER_INDEX_AM; j++)
            {
                if (mag_sums[j] > max_mag)
                {
                    max_mag = mag_sums[j];
                    max_index = j;
                }
            }
            acquire_cfo_adjust(st, max_index - CENTER_AM);
        }

        phase_increment *= cexpf(-sum_xy / sum_x2 * I);
        // TODO: Investigate why 0.06 is needed below
        st->phase *= cexpf((-sum_y / ACQUIRE_SYMBOLS + (sum_xy / sum_x2)*(ACQUIRE_SYMBOLS)*st->fftcp/2 - 0.06) * I);
    }

    for (i = 0; i < ACQUIRE_SYMBOLS; ++i)
    {
        int offset = (st->mode == NRSC5_MODE_FM) ? 0 : (FFT_AM - CP_AM) / 2;
        for (j = 0; j < st->fftcp; ++j)
        {
            float complex sample = st->phase * st->buffer[i * st->fftcp + j + samperr];
            if (j < st->cp)
                st->fftin[(j + offset) % st->fft] = st->shape[j] * sample;
            else if (j < st->fft)
                st->fftin[(j + offset) % st->fft] = sample;
            else
                st->fftin[(j + offset) % st->fft] += st->shape[j] * sample;

            st->phase *= phase_increment;
        }
        st->phase /= cabsf(st->phase);

        fftwf_execute((st->mode == NRSC5_MODE_FM) ? st->fft_plan_fm : st->fft_plan_am);
        fftshift(st->fftout, st->fft);
        sync_push(&st->input->sync, st->fftout);
    }

    keep = st->fftcp + (st->fftcp / 2 - samperr) + st->keep_extra;
    st->keep_extra = 0;
    memmove(&st->in_buffer[0], &st->in_buffer[st->idx - keep], sizeof(float complex) * keep);
    st->idx = keep;
}

void acquire_keep_extra(acquire_t *st, int extra)
{
    st->keep_extra = extra;
}

void acquire_cfo_adjust(acquire_t *st, int cfo)
{
    st->cfo += cfo;
}

unsigned int acquire_push(acquire_t *st, const float complex *buf, const unsigned int length)
{
    const unsigned int size = st->fftcp * (ACQUIRE_SYMBOLS + 1);
    const unsigned int needed = size - st->idx;

    unsigned int pushed = length;
    if (pushed > needed)
        pushed = needed;

    memcpy(&st->in_buffer[st->idx], buf, sizeof(float complex) * pushed);
    st->idx += pushed;

    return pushed;
}

void acquire_reset(acquire_t *st)
{
    firdecim_cf32_reset(st->filter_fm_lower);
    firdecim_cf32_reset(st->filter_am);
    st->idx = 0;
    st->prev_angle = 0;
    st->phase = 1;
    st->keep_extra = 0;
    st->cfo = 0;
}

void acquire_init(acquire_t *st, input_t *input)
{
    int i;

    st->mode = NRSC5_MODE_FM;
    st->fft = FFT_FM;
    st->fftcp = FFTCP_FM;
    st->cp = CP_FM;

    st->input = input;

    st->filter_fm_lower = firdecim_cf32_c_create(filter_lower_taps_fm, sizeof(filter_lower_taps_fm) / sizeof(filter_lower_taps_fm[0]));
    st->filter_fm_upper = firdecim_cf32_c_create(filter_upper_taps_fm, sizeof(filter_upper_taps_fm) / sizeof(filter_upper_taps_fm[0]));

    st->filter_am = firdecim_cf32_create(filter_taps_am, sizeof(filter_taps_am) / sizeof(filter_taps_am[0]));

    pthread_mutex_lock(&fftw_mutex);
    st->fftin = fftwf_alloc_complex(FFT_FM);
    st->fftout = fftwf_alloc_complex(FFT_FM);
    st->fft_plan_fm = fftwf_plan_dft_1d(FFT_FM, st->fftin, st->fftout, FFTW_FORWARD, FFTW_ESTIMATE);
    st->fft_plan_am = fftwf_plan_dft_1d(FFT_AM, st->fftin, st->fftout, FFTW_FORWARD, FFTW_ESTIMATE);
    pthread_mutex_unlock(&fftw_mutex);

    for (i = 0; i < FFTCP_FM; ++i)
    {
        // Pulse shaping window function for FM
        if (i < CP_FM)
            st->shape_fm[i] = sinf(M_PI / 2 * i / CP_FM);
        else if (i < FFT_FM)
            st->shape_fm[i] = 1;
        else
            st->shape_fm[i] = cosf(M_PI / 2 * (i - FFT_FM) / CP_FM);
    }

    for (i = 0; i < FFTCP_AM; ++i)
    {
        // Pulse shaping window function for AM
        if (i < CP_AM)
            st->shape_am[i] = sinf(M_PI / 2 * i / CP_AM);
        else if (i < FFT_AM)
            st->shape_am[i] = 1;
        else
            st->shape_am[i] = cosf(M_PI / 2 * (i - FFT_AM) / CP_AM);
    }

    st->shape = st->shape_fm;

    acquire_reset(st);
}

void acquire_set_mode(acquire_t *st, int mode)
{
    st->mode = mode;

    if (st->mode == NRSC5_MODE_FM)
    {
        st->fft = FFT_FM;
        st->fftcp = FFTCP_FM;
        st->cp = CP_FM;
        st->shape = st->shape_fm;
    }
    else
    {
        st->fft = FFT_AM;
        st->fftcp = FFTCP_AM;
        st->cp = CP_AM;
        st->shape = st->shape_am;
    }
}

void acquire_free(acquire_t *st)
{
    firdecim_cf32_free(st->filter_fm_lower);
    firdecim_cf32_free(st->filter_am);

    pthread_mutex_lock(&fftw_mutex);
    fftwf_destroy_plan(st->fft_plan_fm);
    fftwf_destroy_plan(st->fft_plan_am);
    fftwf_free(st->fftin);
    fftwf_free(st->fftout);
    pthread_mutex_unlock(&fftw_mutex);
}
