#include "naming.hpp"

#include <algorithm>
#include <ranges>

std::optional<uint64_t> Naming::name(auto &&func) const {
    V partition(max_m - min_m + 1, 0);
    for (auto m = min_m; m <= max_m; m++) {
        for (auto i = 0zu; i < arr_sizes[m]; i++) {
            if (func(m, i)) { // i was chosen
                partition[m - min_m]++;
            }
        }
    }
    auto [lb, ub] = std::ranges::equal_range(partitions, partition);
    if (lb == ub)
        return {}; // not a valid partition

    auto nm = 0ull;
    auto partition_id = std::distance(partitions.begin(), lb);
    auto &ms = m_sizes[partition_id];
    for (auto m = min_m; m <= max_m; m++) {
        nm *= ms[m - min_m];
        nm += name_binomial(func, partition[m - min_m], m);
    }
    if (partition_id)
        return nm + partition_sizes_cumsum[partition_id - 1];
    return nm;
}

uint64_t Naming::name_binomial(auto &&func, uint64_t k, uint64_t m) const {
    auto sz = arr_sizes[m];
    auto nm = 0ull;
    for (auto i = 0zu; k; i++) {
        if (func(m, i)) { // i was chosen
            k--;
        } else { // i was not chosen
            // compensate for the names on the 'taken' path
            nm += binomial(sz - 1, k - 1);
        }
        sz--;
    }
    return nm;
}

void Naming::resolve_binomial(auto &&func, uint64_t k, uint64_t m, uint64_t nm) const {
    auto sz = arr_sizes[m];
    for (auto i = 0zu; k; i++) {
        auto mid = binomial(sz - 1, k - 1);
        if (nm < mid) { // i was chosen
            func(m, i);
            k--;
        } else { // i was not chosen
            // skip for the names on the 'taken' path
            nm -= mid;
        }
        sz--;
    }
}

bool Naming::resolve(uint64_t nm, auto &&func) const {
    auto partition_it = std::ranges::upper_bound(partition_sizes_cumsum, nm);
    if (partition_it == partition_sizes_cumsum.end())
        return false; // nm too large
    auto partition_id = partition_it - partition_sizes_cumsum.begin();
    if (partition_id)
        nm -= partition_sizes_cumsum[partition_id - 1];
    auto &msrcp = m_sizes_rcumprod[partition_id];
    for (auto &&[rcp, k, d] : std::views::zip(msrcp, partitions[partition_id], std::views::iota(0ull)))
        resolve_binomial(func, k, min_m + d, nm / rcp), nm %= rcp;
    return true;
}
