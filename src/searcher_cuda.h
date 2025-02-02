#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <type_traits>

#include "growable.h"

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
        bool operator==(const CudaSearcher::R &other) const = default;
    };
    using B = std::type_identity_t<Rg<R>[256]>;

    [[nodiscard]] uint32_t get_height() const { return height; }
    [[nodiscard]] uint64_t size(size_t i) const { return solutions[i].len; }
    [[nodiscard]] uint64_t size() const {
        auto cnt = 0ull;
        for (auto i = 0u; i <= 255u; i++)
            cnt += size(i);
        return cnt;
    }
    [[nodiscard]] uint64_t next_size() const;

    void search_GPU(bool fake = false);

private:
    B solutions;
    uint32_t height; // maximum hight stored in solutions

    void free();
};

bool operator<(const CudaSearcher::R &lhs, const CudaSearcher::R &rhs);

void show_devices();
