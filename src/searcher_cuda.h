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

    struct R {
        uint64_t empty_area;
        uint32_t d;
        uint32_t ex[7];
    };

private:
    R *solutions;
    // n_solutions[0] for device write start
    // n_solutions[1] for device write finish
    volatile uint32_t *n_solutions;
    uint32_t n_solution_processed, n_kernel_invoked;
    volatile uint32_t *n_pending;

    void invoke_kernel(const R &regs);
};

void show_devices();
