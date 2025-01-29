#include <atomic>
#include <print>
#include <cstdlib>
#include <unordered_set>
#include <sys/mman.h>

#include "naming.hpp"
#include "known.hpp"
#include "Shape.hpp"
#include "naming.inl"

struct ISearcher {
    virtual void step(Shape<8> empty_area) = 0;
};

// NOT thread-safe at all!
template <typename Derived, unsigned Sym>
class SearcherCRTP : public ISearcher {
    const Naming &nme;

    std::unordered_set<Shape<8>> used_pieces; // canonical forms

    void step_impl(Shape<8> empty_area) {
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
            sh = sh.canonical_form(Sym);
            if (!used_pieces.insert(sh).second)
                return; // duplicate shape
            step_impl(empty_area - sh);
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
    explicit SearcherCRTP(const Naming &nm) : nme{ nm } { }

public:
    void step(Shape<8> empty_area) override {
        step_impl(empty_area);
    }
};

template <unsigned Sym>
struct AtomicBitSearcher : SearcherCRTP<AtomicBitSearcher<Sym>, Sym> {
    uint64_t *ledger;
    size_t id, width; // bits
    AtomicBitSearcher(uint64_t *ptr, size_t i, size_t w, const Naming &nm)
        : SearcherCRTP<AtomicBitSearcher, Sym>{ nm },
          ledger{ ptr }, id{ i }, width{ w } { }

    void log(uint64_t v) {
        auto stride = (width + 63) / 64;
        std::atomic_ref atm{ ledger[v * stride + id / 64] };
        atm.fetch_or(1ull << (id % 64), std::memory_order_relaxed);
    }
};

template <unsigned Sym>
struct FileByteSearcher : SearcherCRTP<FileByteSearcher<Sym>, Sym> {
    unsigned char *ledger; // mmap(2)
    size_t id, width; // bytes
    FileByteSearcher(void *ptr, size_t i, size_t w, const Naming &nm)
        : SearcherCRTP<FileByteSearcher, Sym>{ nm },
          ledger{ reinterpret_cast<unsigned char *>(ptr) }, id{ i }, width{ w } { }

    void log(uint64_t v) {
        ledger[v * width + id] = 0xff;
    }
};

int main(int argc, char *argv[]) {
    auto sym_C = ::getenv("C") && *::getenv("C");
    auto min_m = std::atoi(argv[1]);
    auto max_m = std::atoi(argv[2]);
    auto min_n = std::atoi(argv[3]);
    auto max_n = std::atoi(argv[4]);
    auto tgt = std::atoi(argv[5]);
    Naming nm{
        (uint64_t)min_m, (uint64_t)max_m,
        (uint64_t)min_n, (uint64_t)max_n, (uint64_t)tgt,
        sym_C ? known_C_shapes : known_shapes,
        sym_C ? shapes_C_count : shapes_count };
    std::print("number of pieces combinations: {}\n", nm.size());
    // ISearcher schr;
}
