#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>

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
    enum mem_t {
        EMPTY = 0,
        HOST = 1,
        DEVICE = 2,
        ARRAY = 3,
        UNIFIED = 4
    };

    [[nodiscard]] uint32_t get_height() const { return height; }
    [[nodiscard]] uint64_t size() const { return n_solutions; }
    [[nodiscard]] uint64_t next_size() const { return n_next; }

    void search_CPU1();
    void search_CPU();
    void search_GPU(mem_t mem);

    [[nodiscard]] mem_t status() const;

private:

    R *solutions;
    uint32_t height; // maximum hight stored in solutions
    uint64_t n_solutions;
    uint64_t n_next;

    void free();
    void ensure_CPU(); // HOST || UNIFIED
    [[nodiscard]] uint64_t next_size(uint64_t i) const;
};

void h_row_search(
        CudaSearcher::R *solutions,
        std::atomic<uint64_t> &n_solutions,
        std::atomic<uint64_t> &n_next,
        CudaSearcher::C cfg);

void show_devices();
