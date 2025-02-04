#include "searcher_cuda.h"

#include <cstring>
#include <print>
#include <iostream>

#define BOOST_THREAD_VERSION 5
#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/thread/future.hpp>
#include <boost/unordered/concurrent_flat_set.hpp>

#include "util.hpp"

struct hasher {
    size_t operator()(const R &v) const {
        size_t h{};
        h |= v.empty_area;
        h ^= (size_t)v.nm_cnt << 0;
        h ^= (size_t)v.ex[0] << 4;
        h ^= (size_t)v.ex[1] << 32 + 4;
        h ^= (size_t)v.ex[2] << 32;
        h ^= (size_t)v.ex[3] << 48;
        return h;
    }
};

struct alignas(64) CSR : boost::concurrent_flat_set<R, hasher> {};

Sorter::Sorter(CudaSearcher &p)
    : parent{ p }, dedup{}, total{},
      pool{ new boost::basic_thread_pool{} },
      sets{ new CSR[256]{} } {
    static_assert(std::alignment_of_v<decltype(sets[0])> >= 64);
}

Sorter::~Sorter() {
    delete pool;
    delete [] sets;
}

void Sorter::join() {
    pool->close();
    pool->join();
    for (auto pos = 0u; pos <= 255u; pos++) {
        auto &set = sets[pos];
        if (!set.size()) {
            parent.write_solution(pos, 0zu);
            continue;
        }
        auto r = parent.write_solution(pos, set.size());
        set.cvisit_all([p=r.ptr](R v) mutable {
            *p++ = v;
        });
        set.clear();
    }
}

void Sorter::push(std::vector<Rg<R>> &&cont) {
    if (cont.empty())
        return;

    for (auto r : cont)
        boost::async(*pool, [this, r]{
            auto [ptr, len] = r;
            auto local = 0zu;
            for (auto i = 0zu; i < len; i++) {
                auto pos = ptr[i].empty_area & 0xffu;
                if (sets[pos].insert(ptr[i]))
                    local++;
            }
            std::atomic_ref atm_d{ dedup };
            std::atomic_ref atm_n{ total };
            auto d = atm_d.fetch_add(local, std::memory_order_relaxed) + local;
            auto n = atm_n.fetch_add(len, std::memory_order_relaxed) + len;
            std::print("sorter: {}/{} = {}B ({:.2f}x)\n",
                    d, n, display(d * sizeof(R)), 1.0 * n / d);
            delete [] ptr;
        });

    cont.clear();
}
