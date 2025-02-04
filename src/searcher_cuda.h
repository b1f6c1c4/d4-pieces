#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <semaphore>
#include <thread>
#include <vector>

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

struct R {
    uint64_t empty_area;
    uint32_t ex[4];
    uint32_t nm_cnt;
    bool operator==(const R &other) const = default;
};

struct CudaSearcher;
struct Sorter {
    explicit Sorter(CudaSearcher &p);

    [[nodiscard]] bool ready() const;
    void push(std::vector<Rg<R>> &&cont);
    void join();

private:
    CudaSearcher &parent;

    std::vector<std::thread> threads;
    mutable std::mutex       mtx;
    std::condition_variable  cv;
    std::condition_variable  cv_push;
    unsigned                 pending; // 0~256 work to be done
    std::vector<Rg<R>>       queue; // __host__, delete []
    uint64_t                 batch;
    bool                     closed;

    void thread_entry(int i, int n);
};

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

private:
    Rg<R> solutions[256];
    std::vector<Growable<R>> grs; // per-GPU

    uint32_t height; // maximum hight stored in solutions

    void free();
};

void show_devices();
