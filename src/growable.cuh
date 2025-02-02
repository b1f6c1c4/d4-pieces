#pragma once

#include <cuda.h>
#include <deque>
#include <functional>
#include <ranges>
#include <vector>

template <typename T>
struct Growable {
    struct R {
        T *ptr;
        size_t len; // number of T
    };

private:
    using value_type = T;

    struct RH : R {
        CUmemGenericAllocationHandle h;
    };

    // in units of T
    std::deque<R> vmaps; // sum(vmaps, &R::len) == reserved
    size_t reserved; // offset + mapped <= reserved
    size_t offset;
    size_t used; // used <= mapped
    // vmaps[0].ptr + offset == maps[0].ptr
    std::deque<RH> maps; // sum(maps, &R::len) == mapped
    size_t mapped;
    std::vector<R> evicted_data;
    size_t evicted;
    size_t chunk; // granularity

    CUmemAllocationProp prop;
    CUmemAccessDesc adesc;

public:
    explicit Growable(size_t max = 0);
    ~Growable();

    static_assert(std::is_trivially_constructible_v<T>, "T not trivially constructible");
    static_assert(std::is_trivially_copyable_v<T>, "T not trivially copyable");

    // return how many T can be written without crash
    [[nodiscard]] size_t risk_free_size() const { return mapped - used; }
    [[nodiscard]] size_t get_used() const { return used; }
    // re-organize all pa mappings s.t. reserved >= offset + new_reserved
    void remap(size_t new_max, bool force = false);
    // mark n of Ts are actually consumed
    void commit(size_t n) { used += n; }
    // free up unused pa
    void compact();
    // allocate a contiguous T[n]
    T *get(size_t n) {
        if (ensure(n))
            return vmaps[0].ptr + used;
        return nullptr;
    }

    // T must have opeartor<
    // returns a unified memory, and the caller needs to free it
    R cpu_merge_sort();

    void mem_stat() const;

    // make sure risk_free_size() >= n, and return the write-start point
    [[nodiscard]] bool ensure(size_t n);

    // copy all useful data from the 0-th pa to evicted_data
    // does NOT free up pa
    void evict1();

    // copy all useful data to evicted_data
    // free up all pa
    void evict_all();

    // remove unused vmap
    void cleanup();
};
