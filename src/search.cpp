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
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <random>
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

// Shape -> piece naming
// use 240MiB of memory to guarantee no hash collision
static constexpr size_t fast_canonical_form_mod = 30878073; // dark magic; don't touch
uint64_t fast_canonical_form[fast_canonical_form_mod]; // bss, zero-initialized
void compute_fast_canonical_form(const Naming &nm, unsigned sym) {
    auto count = 0zu;
    auto ttcount = 0zu;
    auto push_translate = [&](uint64_t nm, Shape<8> t) {
        for (auto row = 0u; row <= t.bottom(); row++)
            for (auto col = 0u; col <= t.right(); col++) {
                ttcount++;
                auto v = t.translate_unsafe(col, row).get_value();
                auto &elem = fast_canonical_form[v % fast_canonical_form_mod];
                // note: hash collision on nm == 0 are undiscoverable
                if (elem && elem != nm)
                    throw std::runtime_error{ "panic: dark magic doesn't work!" };
                elem = nm;
            }
    };
    for (auto m = nm.min_m; m <= nm.max_m; m++) {
        for (auto i = 0zu; i < nm.arr_sizes[m]; i++) {
            count++;
            auto sh = Shape<8>(nm.arr[m][i]);
            for (auto s = sym; auto t : sh.transforms(true)) {
                if (s & 1u)
                    push_translate(nm.name_piece(m, i), t);
                s >>= 1;
            }
        }
    }
    std::print("cached {}/{} => {} canonical forms\n", ttcount, fast_canonical_form_mod, count);
    // the code searching for dark magic
    /*
    std::atomic<uint64_t> best{ ~0ull };
    {
        std::vector<std::jthread> thrs;
        for (auto i = 0; i < 128; i++)
            thrs.emplace_back([&](int seed) {
                std::mt19937_64 rnd{};
                rnd.seed(seed);
                std::uniform_int_distribution<uint64_t> dist(map.size(), 30878073ull);
                std::vector<char> fm(30878073ull, 0);
            bad:
                auto mod = dist(rnd);
                std::memset(fm.data(), 0, mod);
                for (auto [k, v] : map) {
                    if (fm[k.get_value() % mod])
                        goto bad;
                    fm[k.get_value() % mod] = true;
                }
                auto old = best.load();
                if (mod >= old) goto bad;
                while (best.compare_exchange_weak(old, mod, std::memory_order_relaxed))
                    if (mod >= old) goto bad;
                std::print("found: {}/{} => {} canonical forms\n", map.size(), mod, count);
                goto bad;
            }, i);
    }
    std::print("cached {}/{} => {} canonical forms\n", map.size(), fast_canonical_form_mod, count);
    std::abort();
    */
}

static std::atomic_uint64_t g_work_counter, g_board_counter;

// NOT thread-safe at all!
// DO NOT REUSE
template <typename Derived, bool Shortcut = false>
class SearcherCRTP {
    using set_t = boost::container::flat_set<Shape<8>::shape_t>;
    using sv_t = boost::container::small_vector<uint64_t, 24zu>;
    const Naming &nme;
    unsigned sym;
    size_t n_used_pieces;
    std::vector<char> used_pieces; // [nme.name_piece(m,i)] -> bool

public:
    uint64_t step(Shape<8> empty_area) {
        if (!empty_area) {
            if (n_used_pieces < nme.min_n || n_used_pieces > nme.max_n)
                return 0;
            auto id = nme.name([this](uint64_t m, uint64_t i){
                return !!used_pieces[nme.name_piece(m, i)];
            });
            if (id) {
                static_cast<Derived *>(this)->log(*id);
                return 1;
            } else {
                std::print("Warning: Naming rejected SearcherCRTP's plan\n");
                return 0;
            }
        }
        if (n_used_pieces == nme.max_n)
            return 0;

        auto cnt = 0ull;
        auto recurse = [&,this](Shape<8> sh) {
            auto &pcs = used_pieces[fast_canonical_form[sh.get_value() % fast_canonical_form_mod]];
            if (pcs)
                return false; // duplicate shape
            pcs = 1, n_used_pieces++;
            cnt += step(empty_area - sh);
            pcs = 0, n_used_pieces--;
            return cnt && Shortcut;
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
    SearcherCRTP(const Naming &nm, unsigned s)
        : nme{ nm }, sym{ s },
          n_used_pieces{}, used_pieces(nm.size_pieces(), 0) { }
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
        if (!seen.insert(v).second)
            return;
        ledger[v].fetch_add(1, std::memory_order_relaxed);
        g_work_counter.fetch_add(1, std::memory_order_relaxed);
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
        auto mask = 1ull << (id % 64);
        if (!(atm.fetch_or(mask, std::memory_order_relaxed) & mask))
            g_work_counter.fetch_add(1, std::memory_order_relaxed);
    }
};

struct FileSearcherBase {
    static std::pair<int, unsigned char *> open(const char *fn, size_t sz) {
        using namespace std::string_literals;
        auto fd = ::open(fn, O_RDWR | O_CREAT | O_NOATIME,
                S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
        if (fd == -1)
            throw std::runtime_error{ "open(2): "s + std::strerror(errno) };
        if (::lseek(fd, sz, SEEK_SET) == -1)
            throw std::runtime_error{ "lseek(2): "s + std::strerror(errno) };
        {
            char buf{};
            switch (::read(fd, &buf, 1)) {
                case 1:
                    break;
                case -1:
                    throw std::runtime_error{ "read(2): "s + std::strerror(errno) };
                case 0:
                    if (::write(fd, &buf, 1) != 1)
                        throw std::runtime_error{ "write(2): "s + std::strerror(errno) };
                    break;
            }
        }
        if (::lseek(fd, 0, SEEK_SET) == -1)
            throw std::runtime_error{ "lseek(2): "s + std::strerror(errno) };
        auto mem = ::mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mem == MAP_FAILED)
            throw std::runtime_error{ "mmap(2): "s + std::strerror(errno) };
        return { fd, reinterpret_cast<unsigned char *>(mem) };
    }

    static void close(int fd, unsigned char *mem, size_t sz) {
        using namespace std::string_literals;
        if (::munmap(mem, sz))
            throw std::runtime_error{ "mmap(2): "s + std::strerror(errno) };
        if (::close(fd))
            throw std::runtime_error{ "close(2): "s + std::strerror(errno) };
    }
};

struct FileByteSearcher : SearcherCRTP<FileByteSearcher>, private FileSearcherBase {
    unsigned char *ledger; // mmap(2)
    size_t id, width; // bytes
    FileByteSearcher(size_t i, size_t w, const Naming &nm, unsigned s, unsigned char *ptr)
        : SearcherCRTP<FileByteSearcher>{ nm, s },
          ledger{ ptr }, id{ i }, width{ w } { }

    void log(uint64_t v) {
        auto &l = ledger[v * width + id];
        if (!l)
            g_work_counter.fetch_add(1, std::memory_order_relaxed);
        l = 0xff;
    }

    static std::tuple<int, unsigned char *, size_t> open(const char *fn, size_t w, const Naming &nm) {
        auto sz = w * nm.size();
        auto [fd, mem] = FileSearcherBase::open(fn, sz);
        return { fd, mem, sz };
    }

    using FileSearcherBase::close;
};

struct FileBitSearcher : SearcherCRTP<FileBitSearcher>, private FileSearcherBase {
    unsigned char *ledger; // mmap(2)
    size_t id, width; // bits
    size_t mtxs_count;
    std::mutex *mtxs;
    // no copy no move
    FileBitSearcher(size_t i, size_t w, const Naming &nm, unsigned s, unsigned char *ptr)
        : SearcherCRTP<FileBitSearcher>{ nm, s },
          ledger{ ptr }, id{ i }, width{ w }, mtxs_count{ 8 * boost::thread::hardware_concurrency() },
          mtxs{ new std::mutex[mtxs_count]{} } { }
    ~FileBitSearcher() {
        if (mtxs) {
            delete [] mtxs;
            mtxs = nullptr;
        }
    }

    void log(uint64_t v) {
        auto index = v * (width + 7) / 8 + id / 8;
        std::lock_guard lock{ mtxs[index % mtxs_count] };
        auto &l = ledger[index];
        auto mask = 1ull << (id % 8);
        if (!(l & mask)) {
            g_work_counter.fetch_add(1, std::memory_order_relaxed);
            l |= mask;
        }
    }

    static std::tuple<int, unsigned char *, size_t> open(const char *fn, size_t w, const Naming &nm) {
        auto sz = (w + 7) / 8 * nm.size();
        auto [fd, mem] = FileSearcherBase::open(fn, sz);
        return { fd, mem, sz };
    }

    using FileSearcherBase::close;
};

template <typename T, size_t L, typename ... TArgs>
void run(Board<L> board, const Naming &nm, unsigned sym, TArgs && ... args) {
    g_board_counter = 0;
    boost::basic_thread_pool pool;
    board.foreach([&,i=0zu](Shape<8> sh) mutable {
        boost::async(pool, [&,sh](size_t ii) {
            auto cnt = T(ii, board.count, nm, sym, std::forward<TArgs>(args)...).step(sh);
            if (!cnt) {
                std::print("########### ERROR: a board with ZERO cnt found\n{}#######################\n",
                        sh.to_string());
            }
            g_board_counter++;
        }, i++);
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
            std::print("{}/{} board done, {} work done ({}Kpcs/s), {} maximum => {:0.6f}%\n",
                    g_board_counter.load(), bmax,
                    next, (next - old) / 1000.0, max, 100.0 * next / max);
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
    std::print("size to record everything (ab|fb): {} GiB\n", ::pow(2.0, -33) * nm.size() * board.count);
    std::print("size to record everything (fB): {} GiB\n", ::pow(2.0, -30) * nm.size() * board.count);
    if (::getenv("S") == nullptr) {
        std::print("set env S to v|ac|ac|ab|fb|fB\n");
        return 1;
    }

    auto collected = 0ull;
    auto collect = [&](uint64_t i) {
        collected++;
        std::vector<uint64_t> res;
        nm.resolve(i, [&](uint64_t m, uint64_t i) {
            res.push_back(nm.arr[m][i]);
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
    } else if (::getenv("S") == "fb"s) {
        auto [fd, mem, sz] = FileBitSearcher::open(argv[5], board.count, nm);
        std::print("dispatching {} tasks\n", board.count);
        run<FileBitSearcher>(board, nm, sym, mem);
        std::print("closing\n", board.count);
        FileByteSearcher::close(fd, mem, sz);
    } else if (::getenv("S") == "fB"s) {
        auto [fd, mem, sz] = FileByteSearcher::open(argv[5], board.count, nm);
        std::print("dispatching {} tasks\n", board.count);
        run<FileByteSearcher>(board, nm, sym, mem);
        std::print("closing\n", board.count);
        FileByteSearcher::close(fd, mem, sz);
    }

    if (collected)
        std::print("found {}/{} usable results that passed all {} configs!\n",
                collected, nm.size(), board.count);
    else
        std::print("no usable result found from {}, the best could pass {}/{} configs\n",
                nm.size(), best, board.count);
}
