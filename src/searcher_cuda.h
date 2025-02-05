#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <semaphore>
#include <thread>
#include <vector>

#include "record.h"

template <typename T>
struct Rg {
    T *ptr;
    unsigned long long len; // number of T
};

struct CudaSearcher;
namespace boost::executors { class basic_thread_pool; }
struct CSR;
struct Sorter {
    explicit Sorter(CudaSearcher &p);
    ~Sorter();

    void push(Rg<RX> r);
    void join();

private:
    CudaSearcher &parent;
    size_t dedup, total;
    boost::executors::basic_thread_pool *pool;
    CSR *sets;
};

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

    uint32_t height; // maximum hight stored in solutions

    void free();
};

void show_devices();
