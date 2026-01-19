#include "prng.h"

void prng_seed(struct prng* rng, uint64_t init_state, uint64_t init_seq) {
    rng->state = 0;
    rng->inc = (init_seq << 1) | 1;
    prng_rand(rng);

    rng->state += init_state;
    prng_rand(rng);
}

uint32_t prng_rand(struct prng* rng) {
    uint64_t old = rng->state;
    rng->state = old * 6364136223846793005ULL + rng->inc;

    uint32_t xor_shifted = ((old >> 18) ^ old) >> 27;
    uint32_t rot = old >> 59;
    return (xor_shifted >> rot) | (xor_shifted << ((-rot) & 31));
}

static struct prng s_rng = { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL };

void prng_seed_g(uint64_t init_state, uint64_t init_seq) {
    prng_seed(&s_rng, init_state, init_seq);
}

uint32_t prng_rand_g() { return prng_rand(&s_rng); }
