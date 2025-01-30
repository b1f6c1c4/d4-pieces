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

struct CudaSearcher {
    explicit CudaSearcher(size_t num_shapes);
    ~CudaSearcher();

    void start_search(uint64_t empty_area);
    const unsigned char *next();

private:
    unsigned char (*solutions)[8 * 4];
    volatile uint32_t *n_solutions;
    uint32_t n_solution_processed;
    volatile uint32_t *n_pending;
};

void show_devices();
