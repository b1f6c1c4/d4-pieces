#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mimalloc-new-delete.h>
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <mutex>
#include <ranges>
#include <print>
#include <shared_mutex>
#include <semaphore>
#include <sstream>

#define BOOST_THREAD_VERSION 5
#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/thread/future.hpp>

#include "Piece.hpp"
#include "Shape.hpp"
#include "known.hpp"

template <size_t L>
struct board_t {
    Shape<L> base;
    std::vector<Shape<L>> regions;
    size_t count;
};

template <size_t L>
constexpr board_t<L> construct_board(std::string_view sv) {
    std::unordered_map<char, std::string> map;
    auto valid = [](char ch) {
        return (ch >= '0' && ch <= '9')
            || (ch >= 'A' && ch <= 'Z')
            || (ch >= 'a' && ch <= 'z');
    };
    std::string base;
    for (auto ch : sv)
        if (valid(ch))
            map.emplace(ch, "");
    for (auto ch : sv) {
        base.push_back(valid(ch) ? '#' : ch);
        for (auto &[k, v] : map)
            v.push_back(valid(ch) ? k == ch ? '#' : '.' : ch);
    }
    board_t<L> board{
        Shape<L>{ base },
        map | std::views::transform([](auto &kv){ return Shape<L>{ kv.second }; })
            | std::ranges::to<std::vector>(),
        1zu };
    for (auto r : board.regions)
        board.count *= r.size();
    return board;
}

constexpr inline board_t<8> operator ""_b8(const char *str, size_t len) {
    return construct_board<8>({ str, len });
}

constexpr inline board_t<11> operator ""_b11(const char *str, size_t len) {
    return construct_board<11>({ str, len });
}

template <size_t L>
std::optional<Shape<L>> flip(Shape<L> sh) {
    if (auto s = sh.template transform<false, true,  false>(true); s != sh)
        return s;
    if (auto s = sh.template transform<false, false, true >(true); s != sh)
        return s;
    if (auto s = sh.template transform<true,  false, false>(true); s != sh)
        return s;
    if (auto s = sh.template transform<true,  true,  true >(true); s != sh)
        return s;
    return {};
}

using shapes_t = std::vector<Shape<8>>;

template <size_t L>
int exhaustive_search(const board_t<L> &board, bool restrict_C, char *argv[]) {
    using kvp_t = std::pair<Shape<L>, int_fast64_t>;

    auto min_piece_size = std::stoul(argv[1]);
    auto max_piece_size = std::stoul(argv[2]);
    auto min_pieces = std::stoul(argv[3]);
    auto max_pieces = std::stoul(argv[4]);

    std::shared_mutex heap_mtx{};
    std::vector<kvp_t> heap;
    heap.reserve(board.count); // not actually a heap, but sorted array
    [&](this auto &&self, auto it, Shape<L> curr) {
        if (it == board.regions.end()) {
            heap.emplace_back(curr, 0);
            return;
        }
        for (auto pos : *it++)
            self(it, curr.clear(pos));
    }(board.regions.begin(), board.base);

    size_t found_shapes{}, all_shapes{};
    auto valid = ::fcntl(3, F_GETFD) != -1 || errno != EBADF;
    if (valid) {
        std::print("writing to fd:3 ({})\n",
            std::filesystem::read_symlink("/proc/self/fd/3").c_str());
    }
    std::counting_semaphore sem{ 1024z }; // limit memory consumption of pool
    boost::basic_thread_pool pool;
    std::mutex mtx{};
    shapes_t shapes;
    auto report_interval = 10;
    [&](this auto &&self, size_t n, size_t i, size_t pieces, size_t left) -> void {
        if (!left) {
            if (pieces < min_pieces)
                return;
            auto lib = shapes
                | std::views::transform([=](Shape<8> sh){
                      return Piece<L>{ sh.to<L>(), restrict_C ? 0b10010110u : 0b11111111u };
                  })
                | std::ranges::to<std::vector>();
            { // fast-forward, skip check
                std::shared_lock lock{ heap_mtx };
                auto k = heap[0].first;
                lock.unlock();
                if (solve(lib, k, true).empty()) {
                    if (all_shapes++ % report_interval == 0) {
                        std::print("#{:09} ==> total: {}\n", all_shapes, found_shapes);
                        report_interval *= 4;
                    }
                    return;
                }
            }
            sem.acquire();
            boost::async(pool, [&](std::vector<Piece<L>> lib, size_t id) {
                sem.release();
                auto match = 0zu;
                {
                    std::shared_lock lock{ heap_mtx };
                    for (; match < board.count; match++) {
                        auto k = heap[match].first;
                        lock.unlock();
                        auto fail = solve(lib, k, true).empty();
                        lock.lock();
                        std::atomic_ref atm{ heap[match].second };
                        if (fail) {
                            atm.fetch_add(1, std::memory_order_relaxed);
                            break;
                        } else {
                            atm.fetch_sub(1, std::memory_order_relaxed);
                        }
                    }
                }
                {
                    std::lock_guard lock{ mtx };
                    if (match == board.count) {
                        found_shapes++;
                        if (valid) {
                            auto sz = lib.size();
                            ::write(3, &sz, sizeof(sz));
                            for (auto &piece : lib) {
                                auto sh = piece.canonical.template to<8>();
                                ::write(3, &sh, sizeof(Shape<8>));
                            }
                        }
                    }
                    if (id % report_interval == 0) {
                        std::print("#{:09} ==> total: {}\n", id, found_shapes);
                        report_interval *= 4;
                    }
                }
                {
                    std::unique_lock lock{ heap_mtx, std::try_to_lock_t{} };
                    if (!lock.owns_lock())
                        return;
                    std::ranges::stable_sort(
                            std::ranges::subrange{ &heap[0], &heap[board.count] },
                            std::greater{}, &kvp_t::second);
                }
            }, std::move(lib), all_shapes++);
            return;
        }
        while (n <= max_piece_size && i >= shape_count(n))
            n++, i = 0;
        if (n > max_piece_size || left < n || pieces >= max_pieces)
            return;
        auto fork = [&](Shape<8> sh) {
            shapes.push_back(sh);
            self(n, i + 1, pieces + 1, left - n); // taking
            shapes.pop_back();
        };
        fork(shape_at<8>(n, i));
        if (restrict_C) {
            if (auto sh_flip = flip(shape_at<8>(n, i)); sh_flip)
                fork(*sh_flip);
        }
        self(n, i + 1, pieces, left); // not taking
    }(min_piece_size, 0, 0, board.base.size() - board.regions.size());
    pool.close();
    pool.join();

    std::print("found shapes = {}/{}\n", found_shapes, all_shapes);
    return 0;
}

#define L 8

int show_libraries(char *argv[]) {
    auto id = std::stoull(argv[1]);
    for (auto i = 0zu; ; i++) {
        size_t sz;
        if (::read(3, &sz, sizeof(sz)) != sizeof(sz))
            break;
        std::cout.flush();
        std::vector<Piece<L>> pieces;
        pieces.reserve(sz);
        if (id != i)
            ::lseek(3, sz * sizeof(Shape<8>), SEEK_CUR);
        else
            for (auto i = 0zu; i < sz; i++) {
                Shape<8> sh{ 0u };
                ::read(3, &sh, sizeof(sh));
                std::cout << sh.to_string();
            }
    }
    return 0;
}

int main(int argc, char *argv[]) {
    std::stringstream buffer;
    {
        std::ifstream fin(argv++[1]);
        buffer << fin.rdbuf();
    }
    auto board = construct_board<L>(std::string_view(buffer.str()));

    std::print("working on a board of size={} left={} count={}\n",
            board.base.size(), board.base.size() - board.regions.size(), board.count);

    using namespace std::string_view_literals;

    if (argv[1] == "search"sv)
        return exhaustive_search(board, false, ++argv);

    if (argv[1] == "search-C"sv)
        return exhaustive_search(board, true, ++argv);

    if (argv[1] == "show"sv)
        return show_libraries(++argv);
}
