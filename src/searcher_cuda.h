#pragma once

#include <cstdint>
#include <cstddef>

#include "record.h"
#include "sorter.hpp"

struct Device;
struct CudaSearcher {
    explicit CudaSearcher(uint64_t empty_area);
    ~CudaSearcher();

    [[nodiscard]] uint32_t get_height() const { return height; }
    [[nodiscard]] uint64_t size(unsigned pos) const { return solutions[pos].len; }
    [[nodiscard]] uint64_t size() const {
        auto cnt = 0ull;
        for (auto i = 0u; i <= 255u; i++)
            cnt += size(i);
        return cnt;
    }
    [[nodiscard]] uint64_t next_size(unsigned pos) const;
    [[nodiscard]] uint64_t next_size() const {
        auto cnt = 0ull;
        for (auto i = 0u; i <= 255u; i++)
            cnt += next_size(i);
        return cnt;
    }

    void search_GPU();

    Rg<R> write_solution(unsigned pos, size_t sz);
    Rg<R> *write_solutions(size_t sz);

private:
    Rg<R> solutions[256];
    Device *devs;

    uint32_t height; // maximum hight stored in solutions

    void free();
};

void show_devices();
