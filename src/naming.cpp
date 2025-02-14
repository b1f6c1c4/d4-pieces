#include "naming.hpp"

#include <algorithm>
#include <ranges>
#include <numeric>

#include "util.hpp"

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

std::pair<uint64_t, uint64_t> Naming::resolve_piece(uint64_t nm) const {
    auto it = std::ranges::upper_bound(arr_sizes_cumsum, nm);
    if (it == arr_sizes_cumsum.end())
        return { 0, 0 };
    auto id = it - arr_sizes_cumsum.begin();
    if (!id)
        return { min_m, nm };
    return { min_m + id, nm - arr_sizes_cumsum[id - 1] };
}

Naming::Naming(uint64_t m, uint64_t M, uint64_t n, uint64_t N, uint64_t t, const uint64_t * const *a, const size_t *sz)
    : min_m{ m }, max_m{ M }, min_n{ n }, max_n{ N }, tgt{ t }, arr{ a }, arr_sizes{ sz } {
    std::inclusive_scan(arr_sizes + min_m, arr_sizes + max_m + 1,
            std::back_inserter(arr_sizes_cumsum));

    // find partitions
    V curr;
    auto f = [&,that=this](auto &&self, uint64_t left, uint64_t n) {
        auto m = that->min_m + curr.size();
        if (!left) {
            if (n >= that->min_n) {
                curr.resize(that->max_m - that->min_m + 1, 0);
                that->partitions.push_back(curr);
                curr.resize(m - that->min_m, 0);
            }
            return;
        }
        if (m > that->max_m)
            return;
        curr.push_back(0);
        for (auto i = 0zu; i <= left / m && n + i <= that->max_n && i <= that->arr_sizes[m]; i++) {
            curr.back() = i;
            self(self, left - i * m, n + i);
        }
        curr.pop_back();
    };
    f(f, tgt, 0);

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

/* testing

#include "naming.inl"

int main() {
    uint64_t v2[]{ 21, 22, 23 };
    uint64_t v4[]{ 41, 42, 43, 44, 45, 46, 47 };
    uint64_t v5[]{ 51, 52, 53, 54, 55, 56 };
    uint64_t *arr[]{ 0, 0, v2, 0, v4, v5 };
    size_t arr_sz[]{ 0, 0, 3, 0, 7, 6 };
    Naming nm{
        2, 5,
        2, 8,
        20,
        arr,
        arr_sz
    };
    std::print("sz={}\n", nm.size());
    for (auto i = 0ull; i < nm.size(); i++) {
        std::vector<uint64_t> res;
        nm.resolve(i, [&](uint64_t v){res.push_back(v);});
        auto tgt = 0;
        for (auto x : res)
            tgt += x / 10;
        auto v = nm.name([&](uint64_t v){return std::ranges::find(res, v) != res.end();});
        if (20 != tgt || *v != i) {
            std::print("#{}/{} = [ ", i, nm.size());
            for (auto x : res)
                std::print("{}, ", x);
            std::print("] ==> {}\n", *v);
        }
    }
}

*/
