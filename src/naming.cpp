#include "naming.hpp"

#include <algorithm>
#include <ranges>
#include <print>
#include <numeric>

uint64_t Naming::binomial(uint64_t n, uint64_t m) {
    static VV cache;

    if (m > n) return 0;
    while (n >= cache.size()) {
        V next{ 1 };
        if (!cache.empty()) {
            for (auto [p, q] : cache.back() | std::views::adjacent<2>)
                next.push_back(p + q);
            next.push_back(1);
        }
        cache.emplace_back(std::move(next));
    }
    return cache[n][m];
}

Naming::Naming(uint64_t m, uint64_t M, uint64_t n, uint64_t N, uint64_t t, const uint64_t * const *a, const size_t *sz)
    : min_m{ m }, max_m{ M }, min_n{ n }, max_n{ N }, tgt{ t }, arr{ a }, arr_sizes{ sz } {

    // find partitions
    V curr;
    [&,that=this](this auto &&self, uint64_t left, uint64_t n) {
        if (!left) {
            if (n >= that->min_n)
                that->partitions.push_back(curr);
            return;
        }
        auto m = that->min_m + curr.size();
        if (m > that->max_m)
            return;
        curr.push_back(0);
        for (auto i = 0zu; i <= left / m && n + i <= that->max_n && i <= that->arr_sizes[m]; i++) {
            curr.back() = i;
            self(left - i * m, n + i);
        }
        curr.pop_back();
    }(tgt, 0);

    // find m_sizes
    for (auto &pt : partitions) {
        m_sizes.emplace_back();
        for (auto [k, i] : std::views::zip(pt, std::views::iota(0zu)))
            m_sizes.back().push_back(binomial(arr_sizes[min_m + i], k));
        m_sizes_rcumprod.emplace_back();
        std::exclusive_scan(m_sizes.back().rbegin(), m_sizes.back().rend(),
                std::back_inserter(m_sizes_rcumprod.back()), 1ull, std::multiplies{});
        std::ranges::reverse(m_sizes_rcumprod.back());
    }

    // find partition_sizes
    for (auto &ms : m_sizes)
        partition_sizes.push_back(
            std::reduce(ms.begin(), ms.end(),
                1ull, std::multiplies{}));
    std::inclusive_scan(partition_sizes.begin(), partition_sizes.end(),
        std::back_inserter(partition_sizes_cumsum));
}

void Naming::resolve_binomial(std::vector<uint64_t> &out, uint64_t k, uint64_t m, uint64_t nm) const {
    auto sz = arr_sizes[m];
    for (auto i = 0zu; k; i++) {
        auto mid = binomial(sz - 1, k - 1);
        if (nm < mid) { // i was chosen
            out.push_back(arr[m][i]);
            k--;
        } else { // i was not chosen
            nm -= mid;
        }
        sz--;
    }
}

std::vector<uint64_t> Naming::resolve(uint64_t nm) const {
    std::vector<uint64_t> answer;
    auto partition_it = std::ranges::upper_bound(partition_sizes_cumsum, nm);
    if (partition_it == partition_sizes_cumsum.end())
        return {}; // nm too large
    auto partition_id = partition_it - partition_sizes_cumsum.begin();
    if (partition_id)
        nm -= partition_sizes_cumsum[partition_id - 1];
    auto &msrcp = m_sizes_rcumprod[partition_id];
    for (auto &&[rcp, k, d] : std::views::zip(msrcp, partitions[partition_id], std::views::iota(0ull)))
        resolve_binomial(answer, k, min_m + d, nm / rcp), nm %= rcp;
    return answer;
}

int main() {
    uint64_t v2[]{ 21, 22, 23 };
    uint64_t v4[]{ 41, 42, 43, 44, 45, 46 };
    uint64_t v5[]{ 51, 52, 53, 54, 55 };
    uint64_t *arr[]{ 0, 0, v2, 0, v4, v5 };
    size_t arr_sz[]{ 0, 0, 3, 0, 6, 5 };
    Naming nm{
        2, 5,
        2, 8,
        20,
        arr,
        arr_sz
    };
    std::print("sz={}\n", nm.size());
    for (auto i = 0ull; i < nm.size(); i++) {
        auto res = nm.resolve(i);
        auto tgt = 0;
        for (auto x : res)
            tgt += x / 10;
        if (20 == tgt)
            continue;
        std::print("#{}/{} = [ ", i, nm.size());
        for (auto x : res)
            std::print("{}, ", x);
        std::print("]\n");
    }
}
