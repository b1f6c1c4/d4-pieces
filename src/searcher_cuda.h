#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <semaphore>
#include <thread>
#include <vector>

#include <boost/unordered/concurrent_flat_set_fwd.hpp>

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

struct R {
    uint32_t xaL; // requires parse_R and assemble_R
    uint32_t xaH; // requires parse_R and assemble_R
    uint32_t ex0, ex1, ex2;

    bool operator==(const R &other) const = default;
};
static_assert(sizeof(R) == 20);
static_assert(alignof(R) == 4);
struct RX : R {
    uint8_t ea;
};
static_assert(sizeof(RX) == 24);
static_assert(alignof(RX) == 4);

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
