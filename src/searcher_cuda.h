#pragma once

#include <cstddef>
#include <cstdint>

struct tt_t {
    uint64_t shape;
    uint8_t nm;
};
static_assert(sizeof(tt_t) == 16);

// [(Shape, piece naming)]
extern tt_t *fast_canonical_form;
extern size_t fast_canonical_forms;

void fcf_cache(size_t num_shapes);
char *searcher_area(size_t num_shapes);
size_t searcher_step(char *output, uint64_t empty_area);
