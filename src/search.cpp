#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <print>
#include <cstdlib>
#include <stdexcept>
#include <cstring>
#include <stop_token>
#include <thread>
#include <unordered_set>
#include <unordered_map>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <boost/thread/scoped_thread.hpp>

#include "board.hpp"
#include "searcher.hpp"
#include "known.hpp"
#include "Shape.hpp"

struct Verify : SearcherFactory {
    struct S : Searcher {
        Verify &parent;
        explicit S(Verify &p) : Searcher{ true }, parent{ p } { }
        void log(uint64_t) override {
            parent.incr_work();
        }
    };
    Searcher *make() override { return new S{ *this }; };
};

struct AtomicCount : SearcherFactory {
    std::atomic_size_t *ledger;
    struct S : Searcher {
        AtomicCount &parent;
        std::unordered_set<uint64_t> seen;
        explicit S(AtomicCount &p) : parent{ p } { }
        void log(uint64_t v) override {
            if (!seen.insert(v).second)
                return;
            parent.ledger[v].fetch_add(1, std::memory_order_relaxed);
        }
    };
    Searcher *make() override { return new S{ *this }; };
    static uint64_t size() {
        return g_nme->size() * sizeof(std::atomic_size_t) / 8;
    }
};

struct FileSearcherBase : SearcherFactory {
    int fd;
    unsigned char *mem;
    size_t mem_sz;

    FileSearcherBase(const char *fn, size_t sz) : fd{ -1 }, mem{}, mem_sz{ sz } {
        using namespace std::string_literals;
        fd = ::open(fn, O_RDWR | O_CREAT | O_NOATIME,
                S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
        if (fd == -1)
            throw std::runtime_error{ "open(2): "s + std::strerror(errno) };
        if (::lseek(fd, sz, SEEK_SET) == -1)
            throw std::runtime_error{ "lseek(2): "s + std::strerror(errno) };
        { // ensure file is at least sz long
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
        auto m = ::mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (m == MAP_FAILED)
            throw std::runtime_error{ "mmap(2): "s + std::strerror(errno) };
        mem = reinterpret_cast<unsigned char *>(m);
    }

    ~FileSearcherBase() noexcept {
        using namespace std::string_literals;
        if (mem && ::munmap(mem, mem_sz))
            std::print("mmap(2): {}", std::strerror(errno));
        if (fd != -1 && ::close(fd))
            std::print("close(2): {}", std::strerror(errno));
    }
};

struct FileByte : FileSearcherBase {
    FileByte(const char *fn)
        : FileSearcherBase{ fn, g_nme->size() * g_board->count } { }
    struct S : Searcher {
        FileByte &parent;
        explicit S(FileByte &p) : parent{ p } { }
        void log(uint64_t v) {
            auto &l = parent.mem[v * g_board->count + config_index];
            if (!l)
                parent.incr_work();
            l = 0xff;
        }
    };
    Searcher *make() override { return new S{ *this }; };
    static uint64_t size() {
        return g_nme->size() * g_board->count;
    }
};

struct FileBit : FileSearcherBase {
    size_t mtxs_count;
    std::mutex *mtxs;
    // no copy no move
    FileBit(const char *fn)
        : FileSearcherBase{ fn, g_nme->size() * (g_board->count + 7) / 8 },
          mtxs_count{ 8 * boost::thread::hardware_concurrency() },
          mtxs{ new std::mutex[mtxs_count]{} } { }
    ~FileBit() {
        if (mtxs) {
            delete [] mtxs;
            mtxs = nullptr;
        }
    }
    struct S : Searcher {
        FileBit &parent;
        explicit S(FileBit &p) : parent{ p } { }
        void log(uint64_t v) {
            auto index = v * (g_board->count + 7) / 8 + config_index / 8;
            auto mask = 1ull << (config_index % 8);
            std::lock_guard lock{ parent.mtxs[index % parent.mtxs_count] };
            auto &l = parent.mem[index];
            if (!(l & mask)) {
                parent.incr_work();
                l |= mask;
            }
        }
    };
    Searcher *make() override { return new S{ *this }; };
    static uint64_t size() {
        return g_nme->size() * (g_board->count + 7) / 8;
    }
};

std::string display(uint64_t byte) {
    if (byte < 1000ull)
        return std::format("{}", byte);
    if (byte < 1024 * 1024ull)
        return std::format("{:.2f} Ki", 1.0 * byte / 1024);
    if (byte < 1024 * 1024ull * 1024ull)
        return std::format("{:.2f} Mi", 1.0 * byte / 1024 / 1024);
    if (byte < 1024 * 1024ull * 1024ull * 1024ull)
        return std::format("{:.2f} Gi", 1.0 * byte / 1024 / 1024 / 1024);
    return std::format("{:.3f} TiB", 1.0 * byte / 1024 / 1024 / 1024 / 1024);
}

std::jthread monitor(const SearcherFactory &sf) {
    using namespace std::chrono_literals;
    return std::jthread{ [&](std::stop_token st) {
        auto old = 0ull;
        while (!st.stop_requested()) {
            std::this_thread::sleep_for(1s);
            auto next = sf.work_done();
            auto max = g_nme->size() * sf.configs_issued();
            std::print("{}/{}/{} board done, {} work done ({}/s), {} maximum => {:0.6f}%\n",
                    sf.configs_done(), sf.configs_issued(), g_board->count,
                    next, display(next - old),
                    max, 100.0 * next / max);
            old = next;
        }
    } };
}

int main(int argc, char *argv[]) {
    using namespace std::string_literals;

    g_board = Board<8>::from_file(argv++[1]);
    std::print("working on a board of size={} left={} count={}\n",
            g_board->base.size(),
            g_board->base.size() - g_board->regions.size(),
            g_board->count);

    auto sym_C = ::getenv("C") && *::getenv("C");
    g_sym = sym_C ? 0b01101001u : 0b11111111u;
    auto min_m = std::atoi(argv[1]);
    auto max_m = std::atoi(argv[2]);
    auto min_n = std::atoi(argv[3]);
    auto max_n = std::atoi(argv[4]);
    g_nme.emplace(
        (uint64_t)min_m, (uint64_t)max_m,
        (uint64_t)min_n, (uint64_t)max_n,
        g_board->base.size() - g_board->regions.size(),
        sym_C ? known_C_shapes : known_shapes,
        sym_C ? shapes_C_count : shapes_count);
    std::print("number of pieces combinations: {}\n", g_nme->size());
    std::print("total amount of work: {}\n", g_nme->size() * g_board->count);
    std::print("size to record pcsc only (ac): {}\n", display(AtomicCount::size()));
    std::print("size to record everything (fb): {}\n", display(FileBit::size()));
    std::print("size to record everything (fB): {}\n", display(FileByte::size()));

    /*
    auto collected = 0ull;
    auto collect = [&](uint64_t i) {
        collected++;
        std::vector<uint64_t> res;
        g_nme->resolve(i, [&](uint64_t m, uint64_t i) {
            res.push_back(g_nme->arr[m][i]);
        });
        auto sz = res.size();
        ::write(4, &sz, sizeof(sz));
        ::write(4, res.data(), sz * sizeof(uint64_t));
    };
    auto best = 0zu;
    */

    compute_fast_canonical_form();

    std::stop_source stop;

    SearcherFactory *sf;
    if (::getenv("S") == nullptr) {
        std::print("set env S to v|ac|fb|fB\n");
        return 1;
    } else if (::getenv("S") == "v"s) {
        sf = new Verify{};
    } else if (::getenv("S") == "ac"s) {
        sf = new AtomicCount{};
        /* for (auto i = 0ull; i < nm.size(); i++) {
            auto v = ptr[i].load(std::memory_order_relaxed);
            best = std::max(best, v);
            if (v == board.count)
                collect(i);
        } */
    } else if (::getenv("S") == "fb"s) {
        sf = new FileBit{ argv[5] };
    } else if (::getenv("S") == "fB"s) {
        sf = new FileByte{ argv[5] };
    } else {
        std::print("set env S to v|ac|ac|ab|fb|fB\n");
        return 1;
    }

    /*
    if (collected)
        std::print("found {}/{} usable results that passed all {} configs!\n",
                collected, nm.size(), board.count);
    else
        std::print("no usable result found from {}, the best could pass {}/{} configs\n",
                nm.size(), best, board.count);
                */

    auto j = monitor(*sf);
    sf->run();
    delete sf;
}
