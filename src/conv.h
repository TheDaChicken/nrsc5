#ifndef _CONV_H_
#define _CONV_H_

#include <stdint.h>

enum {
	CONV_TERM_FLUSH,
	CONV_TERM_TAIL_BITING,
};

/*
 * Convolutional code descriptor
 *
 * n    - Rate 2, 3, 4 (1/2, 1/3, 1/4)
 * k    - Constraint length (5 or 7)
 * rgen - Recursive generator polynomial in octal
 * gen  - Generator polynomials in octal
 * punc - Puncturing matrix (-1 terminated)
 * term - Termination type (zero flush default)
 */
struct lte_conv_code {
	int n;
	int k;
	int len;
	unsigned rgen;
	unsigned gen[4];
	int *punc;
	int term;
};

/*
 * Trellis Object
 *
 * num_states - Number of states in the trellis
 * sums       - Accumulated path metrics
 * outputs    - Trellis ouput values
 * vals       - Input value that led to each state
 */
struct vtrellis {
	int n;
	int k;
	int num_states;
	int16_t *sums;
	int16_t *outputs;
	uint8_t *vals;
	int intrvl;
};

/*
 * Viterbi Decoder
 *
 * n         - Code order
 * k         - Constraint length
 * len       - Horizontal length of trellis
 * recursive - Set to '1' if the code is recursive
 * intrvl    - Normalization interval
 * trellis   - Trellis object
 * punc      - Puncturing sequence
 * paths     - Trellis paths
 */
struct vdecoder {
	int n;
	int k;
	int len;
	int recursive;
	int *punc;
	int16_t **paths;
};

int conv_alloc_vdec(struct vdecoder *dec, const struct lte_conv_code *code);
void conv_free_paths(struct vdecoder *dec);

int generate_trellis(struct vtrellis *trellis, const struct lte_conv_code *code);
void free_trellis(struct vtrellis *trellis);

int conv_decode(struct vdecoder *vdec, struct vtrellis *trellis, int term, const int8_t *in, uint8_t *out, unsigned int len);

#endif /* _CONV_H_ */
