#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <print>
#include <cstdlib>
#include <stdexcept>
#include <stop_token>
#include <thread>
#include <unordered_set>
#include <sys/mman.h>
#include <mimalloc-new-delete.h>

#define BOOST_THREAD_VERSION 5
#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/thread/future.hpp>

#include "board.hpp"
#include "naming.hpp"
#include "known.hpp"
#include "Shape.hpp"
#include "naming.inl"

std::unordered_map<Shape<8>, Shape<8>> fast_canonical_form;
void compute_fast_canonical_form(const Naming &nm, unsigned sym) {
    auto count = 0zu;
    auto push_translate = [](Shape<8> sh, Shape<8> t) {
        for (auto row = 0u; row <= t.bottom(); row++)
            for (auto col = 0u; col <= t.right(); col++)
                fast_canonical_form.emplace(t.translate_unsafe(col, row), sh);
    };
    for (auto m = nm.min_m; m <= nm.max_m; m++) {
        for (auto i = 0zu; i < nm.arr_sizes[m]; i++) {
            count++;
            auto sh = Shape<8>(nm.arr[m][i]);
            for (auto s = sym; auto t : sh.transforms(true)) {
                if (s & 1u)
                    push_translate(sh, t);
                s >>= 1;
            }
        }
    }
    std::print("cached {} => {} canonical forms\n",
            fast_canonical_form.size(), count);
}

static std::atomic_uint64_t g_work_counter;

// NOT thread-safe at all!
template <typename Derived>
class SearcherCRTP {
    const Naming &nme;
    unsigned sym;
    std::unordered_set<Shape<8>> used_pieces; // canonical forms

public:
    void step(Shape<8> empty_area) {
        if (!empty_area) {
            if (used_pieces.size() < nme.min_n || used_pieces.size() > nme.max_n)
                return;
            auto id = nme.name([this](uint64_t v){
                return used_pieces.contains(Shape<8>{ v });
            });
            if (id) {
                static_cast<Derived *>(this)->log(*id);
                g_work_counter.fetch_add(1, std::memory_order_relaxed);
            } else {
                std::print("Warning: Naming rejected SearcherCRTP's plan\n");
            }
            return;
        }

        auto recurse = [&,this](Shape<8> sh) {
            auto can = fast_canonical_form.at(sh);
            if (!used_pieces.insert(can).second)
                return; // duplicate shape
            step(empty_area - sh);
            used_pieces.erase(can);
        };
        // all possible shapes of size m covering front()
        std::unordered_set<Shape<8>> shapes{ empty_area.front_shape() };
        if (1u >= nme.min_m)
            recurse(empty_area.front_shape());
        for (auto m = 1u; m < nme.max_m; m++) {
            std::unordered_set<Shape<8>> next;
            for (auto sh : shapes) {
                for (auto pos : (sh.extend1() & empty_area) - sh) {
                    next.insert(sh.set(pos));
                }
            }
            if (next.empty())
                break;
            if (m + 1 >= nme.min_m)
                for (auto sh : next)
                    recurse(sh);
            shapes = std::move(next);
        }
    }

protected:
    SearcherCRTP(const Naming &nm, unsigned s) : nme{ nm }, sym{ s } { }
};

struct AtomicCountSearcher : SearcherCRTP<AtomicCountSearcher> {
    std::atomic_size_t *ledger;
    AtomicCountSearcher(size_t, size_t, const Naming &nm, unsigned s, std::atomic_size_t *ptr)
        : SearcherCRTP<AtomicCountSearcher>{ nm, s },
          ledger{ ptr } { }

    void log(uint64_t v) {
        ledger[v].fetch_add(1, std::memory_order_relaxed);
    }
};

struct AtomicBitSearcher : SearcherCRTP<AtomicBitSearcher> {
    uint64_t *ledger;
    size_t id, width; // bits
    AtomicBitSearcher(size_t i, size_t w, const Naming &nm, unsigned s, uint64_t *ptr)
        : SearcherCRTP<AtomicBitSearcher>{ nm, s },
          ledger{ ptr }, id{ i }, width{ w } { }

    void log(uint64_t v) {
        auto stride = (width + 63) / 64;
        std::atomic_ref atm{ ledger[v * stride + id / 64] };
        atm.fetch_or(1ull << (id % 64), std::memory_order_relaxed);
    }
};

struct FileByteSearcher : SearcherCRTP<FileByteSearcher> {
    unsigned char *ledger; // mmap(2)
    size_t id, width; // bytes
    FileByteSearcher(size_t i, size_t w, const Naming &nm, unsigned s, void *ptr)
        : SearcherCRTP<FileByteSearcher>{ nm, s },
          ledger{ reinterpret_cast<unsigned char *>(ptr) }, id{ i }, width{ w } { }

    void log(uint64_t v) {
        ledger[v * width + id] = 0xff;
    }
};

template <typename T, size_t L, typename ... TArgs>
void run(Board<L> board, const Naming &nm, unsigned sym, TArgs && ... args) {
    boost::basic_thread_pool pool;
    board.foreach([&,i=0zu](Shape<8> sh) mutable {
        boost::async(pool, [&](size_t ii) {
            T(ii, board.count, nm, sym, std::forward<TArgs>(args)...).step(sh);
        }, i);
    });
    pool.close();
    pool.join();
};

std::jthread monitor(std::stop_token st, uint64_t max) {
    using namespace std::chrono_literals;
    return std::jthread{ [&] {
        auto old = 0ull;
        while (!st.stop_requested()) {
            std::this_thread::sleep_for(1s);
            auto next = g_work_counter.load(std::memory_order_relaxed);
            std::print("{} done ({}Mipcs/s), {} maximum => {:0.6f}%\n",
                    next, (next - old) / 1024.0 / 1024.0, max, 100.0 * next / max);
            old = next;
        }
    } };
}

int main(int argc, char *argv[]) {
    using namespace std::string_literals;

    auto board = Board<8>::from_file(argv++[1]);
    std::print("working on a board of size={} left={} count={}\n",
            board.base.size(), board.base.size() - board.regions.size(), board.count);

    auto sym_C = ::getenv("C") && *::getenv("C");
    auto sym = sym_C ? 0b01101001u : 0b11111111u;
    auto min_m = std::atoi(argv[1]);
    auto max_m = std::atoi(argv[2]);
    auto min_n = std::atoi(argv[3]);
    auto max_n = std::atoi(argv[4]);
    Naming nm{
        (uint64_t)min_m, (uint64_t)max_m,
        (uint64_t)min_n, (uint64_t)max_n,
        board.base.size() - board.regions.size(),
        sym_C ? known_C_shapes : known_shapes,
        sym_C ? shapes_C_count : shapes_count };
    std::print("number of pieces combinations: {}\n", nm.size());
    std::print("total amount of work: {}\n", nm.size() * board.count);
    std::print("size to record pcsc only (ac): {} GiB\n", ::pow(2.0, -33) * nm.size() * 64);
    std::print("size to record everything (ab): {} GiB\n", ::pow(2.0, -33) * nm.size() * board.count);
    std::print("size to record everything (fB): {} GiB\n", ::pow(2.0, -30) * nm.size() * board.count);
    if (::getenv("S") == nullptr) {
        std::print("set env S to ac|ab|fB\n");
        return 1;
    }

    auto collected = 0ull;
    auto collect = [&](uint64_t i) {
        collected++;
        std::vector<uint64_t> res;
        nm.resolve(i, [&](uint64_t v) {
            res.push_back(v);
        });
        auto sz = res.size();
        ::write(4, &sz, sizeof(sz));
        ::write(4, res.data(), sz * sizeof(uint64_t));
    };

    compute_fast_canonical_form(nm, sym);

    std::stop_source stop;
    auto j = monitor(stop.get_token(), nm.size() * board.count);

    auto best = 0zu;
    if (::getenv("S") == "ac"s) {
        auto ptr = new std::atomic_size_t[nm.size()];
        std::print("dispatching {} tasks\n", board.count);
        run<AtomicCountSearcher>(board, nm, sym, ptr);
        std::print("collecting {} results\n", board.count);
        for (auto i = 0ull; i < nm.size(); i++) {
            auto v = ptr[i].load(std::memory_order_relaxed);
            best = std::max(best, v);
            if (v == board.count)
                collect(i);
        }
        delete [] ptr;
    } else if (::getenv("S") == "ac1"s) {
        auto ptr = new std::atomic_size_t[nm.size()];
        std::print("dispatching {} tasks\n", board.count);
        AtomicCountSearcher(0, 0, nm, sym, ptr).step(board.base.clear(0, 0).clear(3, 3));
        std::print("collecting {} results\n", board.count);
        for (auto i = 0ull; i < nm.size(); i++) {
            auto v = ptr[i].load(std::memory_order_relaxed);
            best = std::max(best, v);
            if (v == board.count)
                collect(i);
        }
        delete [] ptr;
    }
    stop.request_stop();

    if (collected)
        std::print("found {}/{} usable results that passed all {} configs!\n",
                collected, nm.size(), board.count);
    else
        std::print("no usable result found from {}, the best could pass {}/{} configs\n",
                nm.size(), best, board.count);
}
