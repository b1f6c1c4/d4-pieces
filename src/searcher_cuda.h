#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>
#include "growable.cuh"

struct tt_t {
    uint64_t shape;
    uint8_t nm;
};
struct frow_t {
    uint64_t shape;
    union {
        uint8_t nm[4];
        uint32_t nm0123;
    };
};
struct frow_info_t {
    frow_t *data;
    uint32_t sz[6];
};
void frow_cache(const frow_info_t *fiL, const frow_info_t *fiR);

struct CudaSearcher {
    explicit CudaSearcher(uint64_t empty_area);
    ~CudaSearcher();

    struct R {
        uint64_t empty_area;
        uint32_t ex[4];
        uint32_t nm_cnt;
    };
    struct C : R {
        uint32_t height;
    };

    [[nodiscard]] uint32_t get_height() const { return height; }
    [[nodiscard]] uint64_t size() const { return n_solutions; }
    [[nodiscard]] uint64_t next_size() const { return n_next; }
    [[nodiscard]] uint64_t next_size(uint64_t i) const;

    void search_CPU1(bool fake = false);
    void search_CPU(bool fake = false);
    void search_GPU(mem_t mem, bool fake = false);
    void search_Mixed(uint64_t threshold, bool fake = false);

    [[nodiscard]] mem_t status() const;

private:
    Growable<R> solutions[256];
    uint32_t height; // maximum hight stored in solutions
};

void h_row_search(
        CudaSearcher::R *solutions,
        unsigned long long *n_solutions_,
        CudaSearcher::C cfg0);
