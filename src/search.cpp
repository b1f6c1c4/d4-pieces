#include <atomic>
#include <cmath>
#include <cstdint>
#include <print>
#include <cstdlib>
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
            if (auto id = nme.name([this](uint64_t v){
                    return used_pieces.contains(Shape<8>{ v });
                }); id)
                static_cast<Derived *>(this)->log(*id);
            return;
        }

        auto recurse = [&,this](Shape<8> sh) {
            sh = sh.canonical_form(sym);
            if (!used_pieces.insert(sh).second)
                return; // duplicate shape
            step(empty_area - sh);
            used_pieces.erase(sh);
        };
        // all possible shapes of size m covering front()
        std::unordered_set<Shape<8>> shapes{ empty_area.front_shape() };
        if (1u >= nme.min_m)
            recurse(empty_area.front_shape());
        for (auto m = 1u; m <= nme.max_m; m++) {
            std::unordered_set<Shape<8>> next;
            for (auto sh : shapes)
                for (auto pos : (sh.extend1() & empty_area) - sh)
                    next.insert(sh.set(pos));
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
        ::write(3, &sz, sizeof(sz));
        ::write(3, res.data(), sz * sizeof(uint64_t));
    };

    if (::getenv("S") == "ac"s) {
        auto ptr = new std::atomic_size_t[nm.size()];
        std::print("dispatching {} tasks\n", board.count);
        run<AtomicCountSearcher>(board, nm, sym, ptr);
        std::print("collecting {} results\n", board.count);
        for (auto i = 0ull; i < nm.size(); i++) {
            if (ptr[i].load(std::memory_order_relaxed) == board.count)
                collect(i);
        }
        delete [] ptr;
    }

    std::print("found {}/{} usable results that passed all {} configs!\n",
            collected, nm.size(), board.count);
}
