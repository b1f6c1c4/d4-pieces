#pragma once

#include <cstddef>
#include <cstdint>

struct tt_t {
    uint64_t shape;
    uint8_t nm;
};
struct frow_t {
    uint64_t shape;
    uint32_t nm0123;
};
struct frow_info_t {
    const frow_t *data;
    uint32_t sz[5];
};
// ea: 0..15
void frow_cache(unsigned ea, const frow_t *l, const frow_t *r, size_t llen, size_t rlen);

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
    uint32_t n_solution_processed;
    size_t n_kernel_invoked;
    volatile uint32_t *n_pending;

    void invoke_kernel(const R &regs);
};

void show_devices();
