#include "sorter.hpp"

#include <atomic>
#include <barrier>
#include <memory>
#include <mutex>
#include <print>
#include <iostream>
#include <fstream>

#define BOOST_THREAD_VERSION 5
#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/thread/future.hpp>
#include <boost/unordered/concurrent_flat_set.hpp> // 0
#include <boost/unordered/unordered_flat_set.hpp>  // 1
#include <boost/unordered/unordered_set.hpp>       // 2
#include <unordered_set>                           // 3

#include "util.hpp"

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

#if SORTER == 0
struct alignas(64) CSR : boost::concurrent_flat_set<R, hasher> {
};
#elif SORTER == 1
struct alignas(64) CSR : boost::unordered_flat_set<R, hasher> {
    std::mutex mtx;
};
#elif SORTER == 2
struct alignas(64) CSR : boost::unordered_set<R, hasher> {
    std::mutex mtx;
};
#elif SORTER == 3
struct alignas(64) CSR : std::unordered_set<R, hasher> {
    std::mutex mtx;
};
#elif SORTER == 4
struct alignas(64) CSR : boost::unordered_flat_set<R, hasher> {
    std::atomic_flag spin;
};
#endif

Sorter::Sorter()
    : dedup{}, total{}, pending{},
      pool{ new boost::basic_thread_pool{
          boost::thread::hardware_concurrency(),
          [](auto &&) { pthread_setname_np(pthread_self(), "sorter"); }
      } },
#ifdef SORTER_N
      sets{ new CSR[256 * SORTER_N]{} } {
#else
      sets{ new CSR[256]{} } {
#endif
    static_assert(std::alignment_of_v<decltype(sets[0])> >= 64);
}

Sorter::~Sorter() {
    delete pool;
    delete [] sets;
}

std::deque<WL> Sorter::join() {
    pool->close();
    pool->join();
    std::print("sorter: finalizing\n");
#ifndef SORTER_NPARF
    delete pool;
    pool = new boost::basic_thread_pool{
        boost::thread::hardware_concurrency(),
        [](auto &&) { pthread_setname_np(pthread_self(), "sorter-f"); }
    };
#endif
    std::deque<WL> answer(256);
    for (auto pos = 0u; pos <= 255u; pos++) {
#ifdef SORTER_N
        auto sz = 0ull;
        for (auto cnt = 0u; cnt < SORTER_N; cnt++)
            sz += sets[pos * SORTER_N + cnt].size();
#else
        auto sz = sets[pos].size();
#endif
        if (!sz) {
            answer[pos].pos = pos;
            continue;
        }
        WL r{ new R[sz], sz, RgType::DELETE, pos };
#ifndef SORTER_NPARF
        boost::async(*pool, [=,this] {
#endif
        auto ptr = r.ptr;
#ifdef SORTER_N
        for (auto cnt = 0u; cnt < SORTER_N; cnt++) {
            auto &set = sets[pos * SORTER_N + cnt];
#else
            auto &set = sets[pos];
#endif
#if defined(BMARK) && SORTER >= 2 && SORTER <= 3
#ifdef SORTER_N
            std::print("sorter: 0b{:08b}/{:2}={} ({}B)\n",
                    pos, cnt, set.size(), display(set.size() * sizeof(R)));
#else
            std::print("sorter: 0b{:08b}={} ({}B)\n",
                    pos, set.size(), display(set.size() * sizeof(R)));
#endif
#endif
#if SORTER == 0
            set.cvisit_all([&](R v) mutable {
                *ptr++ = v;
            });
#else
            for (auto v : set)
                *ptr++ = v;
#endif
            set.clear();
#ifdef SORTER_N
        }
#endif
        if (ptr != r.ptr + sz)
            throw std::runtime_error{ "cvisit_all faulty" };
#ifndef SORTER_NPARF
        });
#endif
        answer[pos] = r;
    }
#ifndef SORTER_NPARF
    pool->close();
    pool->join();
#endif
    return answer;
}

#define N_PAGES 48
#define N (N_PAGES * 4096ull / sizeof(RX))

void Sorter::push(Rg<RX> r) {
    static_assert(N_PAGES * 4096ull % sizeof(RX) == 0, "RX not aligned to N-page boundry");
    if ((size_t)r.ptr % 4096ull)
        throw std::runtime_error{ "ptr not aligned to page boundry" };
    std::atomic_ref atm_p{ pending };
    atm_p.fetch_add(r.len, std::memory_order_relaxed);
    auto n = (r.len + N - 1) / N;
    auto amount = N;
    auto max_n = 64ull * boost::thread::hardware_concurrency();
    if (n > max_n) {
        amount = ((r.len + max_n - 1) / max_n + N - 1) / N * N;
        n = (r.len + amount - 1) / amount;
    }
    if (n * amount < r.len || n && (n - 1) * amount >= r.len)
        throw std::runtime_error{ std::format("internal error: distributing {} to {} with {} each",
                r.len, n, amount) };
    auto deleter = [=,this] mutable {
        r.dispose();
        std::atomic_ref atm_d{ dedup };
        std::atomic_ref atm_n{ total };
        std::atomic_ref atm_p{ pending };
        atm_p.fetch_sub(r.len, std::memory_order_relaxed);
        auto val_d = atm_d.load(std::memory_order_relaxed);
        auto val_n = atm_n.load(std::memory_order_relaxed);
        std::print("sorter: {}/{} = {}B (+{} = {} x {}) ({:.2f}x)\n",
                val_d, val_n, display(val_d * sizeof(R)),
                r.len, n, amount,
                1.0 * val_n / val_d);
    };
    auto barrier = std::make_shared<std::barrier<decltype(deleter)>>(n, deleter);
    for (auto i = 0u; i < n; i++)
        boost::async(*pool, [=,this](RX *ptr, size_t len) {
            auto local = 0zu;
            for (auto i = 0zu; i < len; i++) {
#ifdef SORTER_N
                auto &set = sets[ptr[i].ea * SORTER_N + ptr[i].ex0 % SORTER_N];
#else
                auto &set = sets[ptr[i].ea];
#endif
#if SORTER == 0
                if (set.insert(ptr[i])) // slice
                    local++;
#else
#if SORTER == 4
                while (set.spin.test_and_set(std::memory_order_acquire));
#else
                std::lock_guard lock{ set.mtx };
#endif
                if (set.insert(ptr[i]).second) // slice
                    local++;
#if SORTER == 4
                set.spin.clear(std::memory_order_release);
#endif
#endif
            }
            std::atomic_ref atm_d{ dedup };
            std::atomic_ref atm_n{ total };
            atm_d.fetch_add(local, std::memory_order_relaxed);
            atm_n.fetch_add(len, std::memory_order_relaxed);
            barrier->arrive_and_drop();
        }, r.ptr + i * amount, std::min(r.len - i * amount, amount));
}

#define C (512ull * 1048576ull)
unsigned Sorter::print_stats() const {
    auto mem_total = 0ull;
    auto mem_free = 0ull;
    for (std::ifstream fin("/proc/meminfo"); fin; ) {
        unsigned long long x;
        std::string s;
        fin >> s;
        if (fin.eof())
            break;
        if (s == "MemTotal:")
            fin >> mem_total, mem_total *= 1024, fin >> s;
        else if (s == "MemFree:")
            fin >> mem_free, mem_free *= 1024, fin >> s;
        else
            fin >> x, fin >> s;
    }

    std::atomic_ref atm_p{ pending };
    auto p = atm_p.load(std::memory_order_relaxed) * sizeof(RX);

    auto sz = 0ull;
    auto bkt = 0ull;
#ifdef SORTER_N
    for (auto i = 0u; i < 256u * SORTER_N; i++) {
#else
    for (auto i = 0u; i < 256u; i++) {
#endif
        auto &set = sets[i];
#if SORTER == 4
        while (set.spin.test_and_set(std::memory_order_acquire));
#elif SORTER > 0
        std::lock_guard lock{ set.mtx };
#endif
        sz += set.size() * sizeof(R);
        bkt += set.bucket_count() * sizeof(R);
#if SORTER == 4
        set.spin.clear(std::memory_order_release);
#endif
    }

#define WIDTH 66
    auto lines = 0u;
    auto w = 0;
    std::stringstream ss;
    auto push = [&](int color, char c) {
        if (++w == 67) {
            ss << "\33[K\33[90m|\n     |";
            w = 1;
            lines++;
        }
        ss << "\33[" << color << 'm' << c;
    };
    ss << "\33[37msys  [";
    for (auto i = 0ull; i < (mem_total - mem_free + C / 2) / C; i++)
        push(31, '|');
    for (auto i = 0ull; i < (mem_free + C / 2) / C; i++)
        push(30, ' ');
    ss << "\33[37m]\33[K\33[0m\n";
    w = 0, lines++;
    ss << "\33[37mmem  [";
    for (auto i = 0ull; i < (p + C / 2) / C ; i++)
        push(95, 'R');
    for (auto i = 0ull; i < (sz + C / 2) / C ; i++)
        push(35, '#');
    for (auto i = (sz + C / 2) / C; i < (bkt + C / 2) / C ; i++)
        push(35, '_');
    ss << "\33[37m]\33[K\33[0m\n";
    w = 0, lines++;
    std::cerr << ss.str();
    return lines;
}
