/* based on pcg rng (https://pcg-random.org)
 * Licensed under Apache License 2.0 (NO WARRANTY, etc. see website) */

/* also based on Magicalbat's implementation
 * https://github.com/Magicalbat/videos/blob/main/rand.c */

#ifndef _PRNG_H
#define _PRNG_H

#include <stdint.h>

struct prng {
    uint64_t state, inc;
};

void prng_seed(struct prng* rng, uint64_t init_state, uint64_t init_seq);
uint32_t prng_rand(struct prng* rng);

void prng_seed_g(uint64_t init_state, uint64_t init_seq);
uint32_t prng_rand_g();

#endif
