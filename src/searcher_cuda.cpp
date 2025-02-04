#include "searcher_cuda.h"

#include <cstring>
#include <print>
#include <unordered_set>
#include <pthread.h>
#include <sched.h>

Sorter::Sorter(CudaSearcher &p)
    : parent{ p },
      threads{}, mtx{}, cv{}, cv_push{}, pending{ 256 },
      queue{}, batch{}, closed{} {
    auto total = std::thread::hardware_concurrency();
    total = std::min(total, 256u);
    auto k = 0;
    for (auto i = 0u; i < total; i++) {
        auto n = i < 256 % total ? (256 + total - 1) / total : 256 / total;
        threads.emplace_back(&Sorter::thread_entry, this, k, n);
        k += n;
    }
    if (k != total)
        throw std::runtime_error{ "internal error" };
}

void Sorter::join() {
    std::unique_lock lock{ mtx };
    closed = true;
    lock.unlock();
    cv.notify_all();

    for (auto &th : threads)
        th.join();
    threads.clear();

    for (auto r : queue)
        delete [] r.ptr;
    queue.clear();
}

namespace std {
    template <>
    struct hash<R> {
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
}

bool Sorter::ready() const {
    std::unique_lock lock{ mtx };
    return !pending;
}

void Sorter::push(std::vector<Rg<R>> &&cont) {
    std::unique_lock lock{ mtx };
    cv_push.wait(lock, [this]{ return pending == 0; });
    pending += 256;
    for (auto r : queue)
        delete [] r.ptr;
    queue = std::move(cont);
    batch++;
    lock.unlock();
    cv.notify_all();
}

void Sorter::thread_entry(int i, int n) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(i, &set);
    if (sched_setaffinity(0, sizeof(set), &set) == -1)
        std::print("sched_setaffinity(2): {}\n", std::strerror(errno));

    std::vector<std::unordered_set<R>> sets(n);

    std::vector<Rg<R>> local_copy;
    uint64_t last{};
    while (true) {
        std::unique_lock lock{ mtx };
        cv.wait(lock, [=,this]{ return batch > last; });
        if (batch > last + 1)
            throw std::runtime_error{ "missing batch" };
        local_copy = queue; // copy
        auto cls = closed;
        lock.unlock();
        for (auto r : local_copy) {
            for (auto &set : sets)
                set.reserve(set.size() + r.len / 200);
            for (auto i = 0ull; i < r.len; i++) {
                auto pos = (r.ptr[i].empty_area & 0xffu);
                if (pos >= i && pos < i + n) [[unlikely]]
                    sets[pos - i].emplace(r.ptr[i]);
            }
        }
        local_copy.clear();
        if (cls)
            break;
        last++;
        lock.lock();
        pending -= n;
        cv_push.notify_one();
    }

    for (auto pos = i; pos < i + n; pos++) {
        auto r = parent.write_solution(pos, sets[i].size());
        for (auto i = 0zu; auto &v : sets[i])
            r.ptr[i++] = v;
    }
}
