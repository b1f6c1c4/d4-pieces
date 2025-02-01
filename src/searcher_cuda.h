#pragma once

#include <cstddef>
#include <cstdint>

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
    };
    struct C : R {
        uint32_t height;
    };

    [[nodiscard]] uint64_t size() const { return n_solutions; }
    [[nodiscard]] uint64_t next_size() const { return n_next; }

    void search_CPU1();
    void search_CPU();
    void search_GPU();

private:
    enum mem_t {
        EMPTY = 0,
        HOST = 1,
        DEVICE = 2,
        ARRAY = 3,
        UNIFIED = 4
    };
    [[nodiscard]] mem_t status() const;

    R *solutions;

    uint32_t height;
    uint64_t n_solutions;
    uint64_t n_next;

    void ensure_CPU();
    void ensure_GPU();
};

void show_devices();
