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
#include <unordered_map>
#include <sys/mman.h>
#include <mimalloc-new-delete.h>

#define BOOST_THREAD_VERSION 5
#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/thread/future.hpp>
#include <boost/container/flat_set.hpp>
#include <boost/container/small_vector.hpp>

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

static std::atomic_uint64_t g_work_counter, g_board_counter;

// NOT thread-safe at all!
template <typename Derived, bool Shortcut = false>
class SearcherCRTP {
    using set_t = boost::container::flat_set<Shape<8>::shape_t>;
    using sv_t = boost::container::small_vector<uint64_t, 24zu>;
    const Naming &nme;
    unsigned sym;
    set_t used_pieces; // canonical forms

public:
    uint64_t step(Shape<8> empty_area) {
        if (!empty_area) {
            if (used_pieces.size() < nme.min_n || used_pieces.size() > nme.max_n)
                return 0;
            auto id = nme.name([this](uint64_t v){
                return used_pieces.contains(v);
            });
            if (id) {
                static_cast<Derived *>(this)->log(*id);
                g_work_counter.fetch_add(1, std::memory_order_relaxed);
                return 1;
            } else {
                std::print("Warning: Naming rejected SearcherCRTP's plan\n");
                return 0;
            }
        }
        if (used_pieces.size() == nme.max_n)
            return 0;

        auto cnt = 0ull;
        auto recurse = [&,this](Shape<8> sh) {
            auto can = fast_canonical_form.at(sh);
            if (!used_pieces.insert(can.get_value()).second)
                return false; // duplicate shape
            cnt += step(empty_area - sh);
            if (cnt && Shortcut)
                return true;
            used_pieces.erase(can.get_value());
            return false;
        };
        // all possible shapes of size m covering front()
        sv_t shapes{ empty_area.front_shape().get_value() };
        if (1u >= nme.min_m)
            if (recurse(empty_area.front_shape()))
                return cnt;
        for (auto m = 1u; m < nme.max_m; m++) {
            sv_t next;
            next.reserve(4 * shapes.size());
            for (auto shv : shapes) {
                Shape<8> sh{ shv };
                // the following is equivalent to this, but is faster:
                //   for (auto pos : (sh.extend1() & empty_area) - sh)
                //       next.push_back(sh.set(pos).get_value());
                auto ex = (sh.extend1() & empty_area) - sh;
                for (auto v = ex.get_value(); v; v -= (v & -v)) [[likely]]
                    next.push_back(shv | (v & -v));
            }
            if (next.empty())
                break;
            std::ranges::sort(next);
            next.erase(std::ranges::unique(next).begin(), next.end());
            if (m + 1 >= nme.min_m)
                for (auto sh : next)
                    if (recurse(Shape<8>{ sh }))
                        return cnt;
            shapes = std::move(next);
        }
        return cnt;
    }

protected:
    SearcherCRTP(const Naming &nm, unsigned s) : nme{ nm }, sym{ s } {
        used_pieces.reserve(nm.max_n);
    }
};

struct VerifySearcher : SearcherCRTP<VerifySearcher, true> {
    VerifySearcher(size_t, size_t, const Naming &nm, unsigned s)
        : SearcherCRTP<VerifySearcher, true>{ nm, s } { }

    void log(uint64_t v) { }
};

struct AtomicCountSearcher : SearcherCRTP<AtomicCountSearcher> {
    std::atomic_size_t *ledger;
    std::unordered_set<uint64_t> seen;
    AtomicCountSearcher(size_t, size_t, const Naming &nm, unsigned s, std::atomic_size_t *ptr)
        : SearcherCRTP<AtomicCountSearcher>{ nm, s },
          ledger{ ptr } { }

    void log(uint64_t v) {
        if (seen.insert(v).second)
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
    g_board_counter = 0;
    boost::basic_thread_pool pool;
    board.foreach([&,i=0zu](Shape<8> sh) mutable {
        boost::async(pool, [&](size_t ii) {
            auto cnt = T(ii, board.count, nm, sym, std::forward<TArgs>(args)...).step(sh);
            if (!cnt) {
                std::print("########### ERROR: a board with ZERO cnt found\n{}#######################\n",
                        sh.to_string());
            }
            g_board_counter++;
        }, i);
    });
    pool.close();
    pool.join();
};

std::jthread monitor(uint64_t bmax, uint64_t max) {
    using namespace std::chrono_literals;
    return std::jthread{ [=](std::stop_token st) {
        auto old = 0ull;
        while (!st.stop_requested()) {
            std::this_thread::sleep_for(1s);
            auto next = g_work_counter.load(std::memory_order_relaxed);
            std::print("{}/{} board done, {} work done ({}Mipcs/s), {} maximum => {:0.6f}%\n",
                    g_board_counter.load(), bmax,
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
        std::print("set env S to v|ac|ac|ab|fB\n");
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
    auto j = monitor(board.count, nm.size() * board.count);

    auto best = 0zu;
    if (::getenv("S") == "v"s) {
        std::print("dispatching {} scanning tasks\n", board.count);
        run<VerifySearcher>(board, nm, sym);
    } else if (::getenv("S") == "ac"s) {
        auto ptr = new std::atomic_size_t[nm.size()]{};
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
    }

    if (collected)
        std::print("found {}/{} usable results that passed all {} configs!\n",
                collected, nm.size(), board.count);
    else
        std::print("no usable result found from {}, the best could pass {}/{} configs\n",
                nm.size(), best, board.count);
}
