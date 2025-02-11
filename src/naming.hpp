#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

// partition tgt into at least min_n, at most max_n parts
// s.t. each part satisfies min_m <= part <= max_m;
// order each of such partition from smallest to largest
//
// given a partition tgt = k0 * min_m + k1 * (min_m + 1) + ... + k$ * max_m
// for each way of choosing k0 objects from arr[min_m], k1 from ..., k$ from arr[max_m]
// count 1
struct Naming {
    // NOT thread-safe as it modifies binomial::cache
    Naming(uint64_t m, uint64_t M, uint64_t n, uint64_t N, uint64_t t, const uint64_t * const *a, const size_t *sz);

    // thread-safe, O(1)
    [[nodiscard]] uint64_t name_piece(uint64_t m, uint64_t i) const {
        if (m == min_m)
            return i;
        return arr_sizes_cumsum[m - min_m - 1] + i;
    }

    // thread-safe, O(log m)
    [[nodiscard]] std::pair<uint64_t, uint64_t> resolve_piece(uint64_t nm) const;

    // thread-safe, O(1)
    [[nodiscard]] uint64_t size_pieces() const {
        return arr_sizes_cumsum.back();
    }

    // thread-safe, O(1)
    [[nodiscard]] uint64_t size() const {
        if (partition_sizes_cumsum.empty())
            return 0;
        return partition_sizes_cumsum.back();
    }

    // thread-safe, O(sum(arr_sizes) * func)
    // bool func(uint64_t m, uint64_t i) 
    //   shall return true iff the i-th piece of size m was used
    [[nodiscard]] std::optional<uint64_t> name(auto &&func) const;

    // thread-safe, O(sum(arr_sizes) * func)
    // void func(uint64_t m, uint64_t i)
    bool resolve(uint64_t nm, auto &&func) const;

    uint64_t min_m, max_m, min_n, max_n, tgt;
    const uint64_t * const *arr; // not actually used by Naming
    const size_t *arr_sizes;

private:
    using V = std::vector<uint64_t>;
    using VV = std::vector<V>;

    // cache[n][m] == C(n,m) == C(n,n-m)
    static uint64_t binomial(uint64_t n, uint64_t m);

    // find the nm-th way of choosing k out of arr_size[m] objects
    uint64_t name_binomial(auto &&func, uint64_t k, uint64_t m) const;
    void resolve_binomial(auto &&func, uint64_t k, uint64_t m, uint64_t nm) const;

    V arr_sizes_cumsum; // inclusive, starting from min_m
    VV partitions; // tgt == sum((min_m + d) * partitions[*][d], d = 0...max_m-min_m)
    V partition_sizes; // partition_sizes[*] = product(m_sizes[*][d], d = 0...max_m-min_m)
    V partition_sizes_cumsum; // inclusive
    VV m_sizes; // m_sizes[*][d] == C(arr_sizes[min_m+d], partitions[*][d])
    VV m_sizes_rcumprod; // exclusive, reversed
};
