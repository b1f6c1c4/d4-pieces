#include "searcher_cuda.h"

#include <cstring>
#include <print>
#include <iostream>
#include <unordered_set>
#include <pthread.h>
#include <sched.h>

Sorter::Sorter(CudaSearcher &p)
    : parent{ p },
      total{ std::min(std::thread::hardware_concurrency(), 256u) },
      threads{}, mtx{}, cv{}, cv_push{}, pending{ 0 },
      queue{}, batch{}, closed{} {
    auto k = 0;
    for (auto i = 0u; i < total; i++) {
        auto n = i < 256 % total ? (256 + total - 1) / total : 256 / total;
        threads.emplace_back(&Sorter::thread_entry, this, k, n);
        k += n;
    }
    if (k != 256)
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
    if (cont.empty())
        return;
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

void Sorter::thread_entry(int pos, int n) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(pos / 2, &set); // hack
    if (sched_setaffinity(0, sizeof(set), &set) == -1)
        std::print("sched_setaffinity(2): {}\n", std::strerror(errno));

    std::vector<std::unordered_set<R>> sets(n);

    std::vector<size_t> count(n, 0zu);
    auto total = 0ull;
    std::vector<Rg<R>> local_copy;
    uint64_t last{};
    while (true) {
        std::unique_lock lock{ mtx };
        cv.wait(lock, [=,this]{ return batch > last || closed; });
        if (batch > last + 1)
            throw std::runtime_error{ "missing batch" };
        if (batch > last)
            local_copy = queue; // copy
        auto cls = closed;
        lock.unlock();
        for (auto r : local_copy) {
            for (auto &set : sets)
                set.reserve(set.size() + r.len / 200);
            for (auto i = 0ull; i < r.len; i++) {
                auto p = (r.ptr[i].empty_area & 0xffu);
                if (p >= pos && p < pos + n) [[unlikely]] {
                    sets[p - pos].emplace(r.ptr[i]);
                    count[p - pos]++;
                }
            }
            total += r.len;
        }
        local_copy.clear();
        if (cls)
            break;
        last++;
        lock.lock();
        pending -= n;
        cv_push.notify_one();
    }

    for (auto p = pos; p < pos + n; p++) {
        auto &set = sets[p - pos];
        if (!set.size()) {
            parent.write_solution(p, 0zu);
            continue;
        }
        std::print(std::cerr, "sorter#0x{:02x}: {:8}/{:9}/{} sorted ({:.01f}x)\n", p,
                set.size(), count[p - pos], total, 1.0 * count[p - pos] / set.size());
        auto r = parent.write_solution(p, set.size());
        for (auto i = 0zu; auto &v : set)
            r.ptr[i++] = v;
    }
}
