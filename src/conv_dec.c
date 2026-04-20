/*
 * Viterbi decoder for convolutional codes
 *
 * Copyright (C) 2015 Ettus Research LLC
 *
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
 *
 * Author: Tom Tsou <tom.tsou@ettus.com>
 */

#include "config.h"

#include <stdlib.h>
#ifndef __APPLE__
#include <malloc.h>
#endif
#include <string.h>
#include <assert.h>
#include <errno.h>

#include "defines.h"
#include "conv.h"

#include "conv_gen.h"
#if defined(HAVE_SSE3)
#include "conv_sse.h"
#elif defined(HAVE_NEON)
#include "conv_neon.h"
#endif

#define PARITY(X) __builtin_parity(X)
#define TAIL_BITING_EXTRA 32

/*
 * Trellis State
 *
 * state - Internal shift register value
 * prev  - Register values of previous 0 and 1 states
 */
struct vstate {
	unsigned state;
	unsigned prev[2];
};

/*
 * Aligned Memory Allocator
 *
 * SSE requires 16-byte memory alignment. We store relevant trellis values
 * (accumulated sums, outputs, and path decisions) as 16 bit signed integers
 * so the allocated memory is casted as such.
 */
#define SSE_ALIGN	16

static int16_t *vdec_malloc(size_t n)
{
#if defined(HAVE_SSE3) && !defined(__APPLE__)
	return (int16_t *) memalign(SSE_ALIGN, sizeof(int16_t) * n);
#else
	return (int16_t *) malloc(sizeof(int16_t) * n);
#endif
}

/* Left shift and mask for finding the previous state */
static unsigned vstate_lshift(unsigned reg, int k, int val)
{
	unsigned mask;

	if (k == 5)
		mask = 0x0e;
	else if (k == 7)
		mask = 0x3e;
	else if (k == 9)
		mask = 0xfe;
	else
		mask = 0;

	return ((reg << 1) & mask) | val;
}

/*
 * Populate non-recursive trellis state
 *
 * For a given state defined by the k-1 length shift register, find the
 * value of the input bit that drove the trellis to that state. Then
 * generate the N outputs of the generator polynomial at that state.
 */
static void gen_state_info(const struct lte_conv_code *code,
			   uint8_t *val, unsigned reg, int16_t *out)
{
	int i;
	unsigned prev;

	/* Previous '0' state */
	prev = vstate_lshift(reg, code->k, 0);

	/* Compute output and unpack to NRZ */
	*val = (reg >> (code->k - 2)) & 0x01;
	prev = prev | (unsigned) *val << (code->k - 1);

	for (i = 0; i < code->n; i++)
		out[i] = PARITY(prev & code->gen[i]) * 2 - 1;
}

/*
 * Populate recursive trellis state
 */
static void gen_rec_state_info(const struct lte_conv_code *code,
			                   uint8_t *val, unsigned reg, int16_t *out)
{
	int i;
	unsigned prev, rec, mask;

	/* Previous '0' and '1' states */
	prev = vstate_lshift(reg, code->k, 0);

	/* Compute recursive input value (not the value shifted into register) */
	rec = (reg >> (code->k - 2)) & 0x01;

	if ((unsigned int) PARITY(prev & code->rgen) == rec)
		*val = 0;
	else
		*val = 1;

	/* Compute outputs and unpack to NRZ */
	prev = prev | rec << (code->k - 1);

	if (code->k == 5)
		mask = 0x0f;
	else if (code->k == 7)
		mask = 0x3f;
	else
	  mask = 0xff;

	/* Check for recursive outputs */
	for (i = 0; i < code->n; i++) {
		if (code->gen[i] & mask)
			out[i] = PARITY(prev & code->gen[i]) * 2 - 1;
		else
			out[i] = *val * 2 - 1;
	}
}

/* Release the trellis */
void free_trellis(struct vtrellis *trellis)
{
	if (!trellis)
		return;

	free(trellis->vals);
	free(trellis->outputs);
	free(trellis->sums);
}

#define NUM_STATES(K)	(K == 9 ? 256 : (K == 7 ? 64 : 16))


static int conv_decode_length(const int term, const int code_len, const int k)
{
	int out;

	if (term == CONV_TERM_FLUSH)
		out = code_len + k - 1;
	else
		out = code_len + TAIL_BITING_EXTRA * 2;

	return out;
}


/*
 * Allocate and initialize the trellis object
 *
 * Initialization consists of generating the outputs and output value of a
 * given state. Due to trellis symmetry, only one of the transition paths
 * is used by the butterfly operation in the forward recursion, so only one
 * set of N outputs is required per state variable.
 */
int generate_trellis(struct vtrellis *trellis, const struct lte_conv_code *code)
{
	int i;
	int16_t *out;

	const int ns = NUM_STATES(code->k);
	const int olen = (code->n == 2) ? 2 : 4;

	trellis->num_states = ns;
	trellis->sums =	vdec_malloc(ns);
	trellis->outputs = vdec_malloc(ns * olen);
	trellis->vals = (uint8_t *) malloc(ns * sizeof(uint8_t));
	trellis->k = code->k;
	trellis->n = code->n;
	trellis->intrvl = INT16_MAX / (trellis->n * INT8_MAX) - trellis->k;

	if (!trellis->sums || !trellis->outputs || !trellis->vals)
		goto fail;

	/* Populate the trellis state objects */
	for (i = 0; i < ns; i++) {
		out = &trellis->outputs[olen * i];

		if (code->rgen)
			gen_rec_state_info(code, &trellis->vals[i], i, out);
		else
			gen_state_info(code, &trellis->vals[i], i, out);
	}

	return 0;
fail:
	return -1;
}

/*
 * Reset decoder
 *
 * Set accumulated path metrics to zero. For termination other than
 * tail-biting, initialize the zero state as the encoder starting state.
 * Intialize with the maximum accumulated sum at length equal to the
 * constraint length.
 */
static void reset_trellis(const struct vtrellis* trellis, const int term)
{
	const int ns = trellis->num_states;

	memset(trellis->sums, 0, sizeof(int16_t) * ns);

	if (term != CONV_TERM_TAIL_BITING)
		trellis->sums[0] = INT8_MAX * trellis->n * trellis->k;
}

static int _traceback(const struct vdecoder *dec,
		              const struct vtrellis *trellis, unsigned state, uint8_t *out, int len, int offset)
{
	int i;
	unsigned path;

	for (i = len - 1; i >= 0; i--) {
		path = dec->paths[i + offset][state] + 1;
		out[i] = trellis->vals[state];
		state = vstate_lshift(state, trellis->k, path);
	}

	return state;
}

static void _traceback_rec(const struct vdecoder *dec,
			               const struct vtrellis *trellis,
			               unsigned state, uint8_t *out, int len)
{
	int i;
	unsigned path;

	for (i = len - 1; i >= 0; i--) {
		path = dec->paths[i][state] + 1;
		out[i] = path ^ trellis->vals[state];
		state = vstate_lshift(state, trellis->k, path);
	}
}

/*
 * Traceback and generate decoded output
 *
 * For tail biting, find the largest accumulated path metric at the final state
 * followed by two trace back passes. For zero flushing the final state is
 * always zero with a single traceback path.
 */
static int traceback(struct vdecoder *dec, const struct vtrellis *trellis, uint8_t *out, int term, const int len)
{
	int i, sum, max_p = -1, max = -1;
	unsigned path, state = 0;

	const int dec_len = conv_decode_length(term, len, trellis->k);
	if (dec_len > dec->len)
	{
		log_error("Conv decoder length %d is too short for term %d and code length %d", dec->len, term, len);
		return 0;
	}

	if (term == CONV_TERM_TAIL_BITING) {
		for (i = 0; i < trellis->num_states; i++) {
			sum = trellis->sums[i];
			if (sum > max) {
				max_p = max;
				max = sum;
				state = i;
			}
		}
		if (max < 0)
			return -EPROTO;
		for (i = dec_len - 1; i >= len + TAIL_BITING_EXTRA; i--) {
			path = dec->paths[i][state] + 1;
			state = vstate_lshift(state, trellis->k, path);
		}
	} else {
		for (i = dec_len - 1; i >= len; i--) {
			path = dec->paths[i][state] + 1;
			state = vstate_lshift(state, trellis->k, path);
		}
	}

	if (dec->recursive)
		_traceback_rec(dec, trellis, state, out, len);
	else
		state =_traceback(dec, trellis, state, out, len, term == CONV_TERM_TAIL_BITING ? TAIL_BITING_EXTRA : 0);

	/* Don't handle the odd case of recursize tail-biting codes */

	return max - max_p;
}

/* Release decoder object */
void conv_free_paths(struct vdecoder *dec)
{
	if (!dec)
		return;

	if (dec->paths)
		free(dec->paths[0]);
	free(dec->paths);
}

/*
 * Allocate decoder object
 *
 * Subtract the constraint length K on the normalization interval to
 * accommodate the initialization path metric at state zero.
 */
int conv_alloc_vdec(struct vdecoder *dec, const struct lte_conv_code *code)
{
	int i;
	int ns = NUM_STATES(code->k);

	dec->n = code->n;
	dec->k = code->k;
	dec->recursive = code->rgen ? 1 : 0;

    assert(dec->k == 7 || dec->k == 9);

	dec->len = conv_decode_length(code->term, code->len, dec->k);
	dec->paths = (int16_t **) malloc(sizeof(int16_t*) * dec->len);
	if (!dec->paths)
		goto fail;
	dec->paths[0] = vdec_malloc(ns * dec->len);
	if (!dec->paths[0])
		goto fail;
	for (i = 1; i < dec->len; i++)
		dec->paths[i] = &dec->paths[0][i * ns];
	return 0;
fail:
	conv_free_paths(dec);
	return -1;
}

/*
 * Forward trellis recursion
 *
 * Generate branch metrics and path metrics with a combined function. Only
 * accumulated path metric sums and path selections are stored. Normalize on
 * the interval specified by the decoder.
 */
static void _conv_decode(struct vdecoder *dec, struct vtrellis *trellis, const int8_t *seq, const int term, const int len)
{
	int i, j = 0;

	if (term == CONV_TERM_TAIL_BITING)
		j = len - TAIL_BITING_EXTRA;

	if (trellis->k != dec->k)
		return;

	for (i = 0; i < conv_decode_length(term, len, trellis->k); i++, j++) {
		if (term == CONV_TERM_TAIL_BITING && j == len)
			j = 0;

		if (trellis->k == 7)
			gen_metrics_k7_n3(&seq[trellis->n * j],
					 trellis->outputs,
					 trellis->sums,
					 dec->paths[i],
					 !(i % trellis->intrvl));
		else if (trellis->k == 9)
			gen_metrics_k9_n3(&seq[trellis->n * j],
					 trellis->outputs,
					 trellis->sums,
					 dec->paths[i],
					 !(i % trellis->intrvl));
	}
}

int conv_decode(struct vdecoder *vdec, struct vtrellis *trellis, const int term, const int8_t *in, uint8_t *out, const unsigned int len)
{
	reset_trellis(trellis, term);

	/* Propagate through the trellis with interval normalization */
	_conv_decode(vdec, trellis, in, term, len);

	return traceback(vdec, trellis, out, term, len);
}
