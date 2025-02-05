#include "sorter.hpp"

#include <atomic>
#include <print>
#include <iostream>

#define BOOST_THREAD_VERSION 5
#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/thread/future.hpp>
#include <boost/unordered/concurrent_flat_set.hpp>

#include "util.hpp"
#include "searcher_cuda.h"

struct hasher {
    size_t operator()(const R &v) const {
        size_t h{};
        boost::hash_combine(h, v.xaL);
        boost::hash_combine(h, v.xaH);
        boost::hash_combine(h, v.ex0);
        boost::hash_combine(h, v.ex1);
        boost::hash_combine(h, v.ex2);
        return h;
    }
};

struct alignas(64) CSR : boost::concurrent_flat_set<R, hasher> {
};

Sorter::Sorter(CudaSearcher &p)
    : parent{ p }, dedup{}, total{},
      pool{ new boost::basic_thread_pool{} },
      sets{ new CSR[256][17]{} } {
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
        auto sz = 0ull;
        for (auto cnt = 0u; cnt <= 16u; cnt++)
            sz += sets[pos][cnt].size();
        if (!sz) {
            parent.write_solution(pos, 0zu);
            continue;
        }
        auto [ptr, _] = parent.write_solution(pos, sz);
        for (auto cnt = 0u; cnt <= 16u; cnt++) {
            auto &set = sets[pos][cnt];
            set.cvisit_all([&](R v) mutable {
                *ptr++ = v;
            });
            set.clear();
        }
    }
}

void Sorter::push(Rg<RX> r, unsigned height) {
    boost::async(*pool, [this, r, height]{
        auto [ptr, len] = r;
        auto local = 0zu;
        for (auto i = 0zu; i < len; i++)
            if (sets[ptr[i].ea][ptr[i].get_cnt(height)].insert(ptr[i])) // slice
                local++;
        std::atomic_ref atm_d{ dedup };
        std::atomic_ref atm_n{ total };
        auto d = atm_d.fetch_add(local, std::memory_order_relaxed) + local;
        auto n = atm_n.fetch_add(len, std::memory_order_relaxed) + len;
        std::print("sorter: {}/{} = {}B ({:.2f}x)\n",
                d, n, display(d * sizeof(R)), 1.0 * n / d);
        delete [] ptr;
    });
}
