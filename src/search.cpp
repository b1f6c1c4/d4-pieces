#include <algorithm>
#include <atomic>
#include <filesystem>
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <ranges>
#include <print>

#define BOOST_THREAD_VERSION 5
#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/thread/future.hpp>

#include "Piece.hpp"
#include "Shape.hpp"
#include "known.hpp"

struct stat_t {
    size_t zeros, singles;
    size_t min, max;
    double avg;
};

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
bool screen(const std::vector<Piece<L>> &lib, board_t<L> board) {
    auto b = board.base;
    for (auto r : board.regions)
        b = b.clear(r.front());
    if (solve(lib, b, true).empty())
        return false;

    b = board.base;
    for (auto r : board.regions)
        b = b.clear(r.back());
    if (solve(lib, b, true).empty())
        return false;

    return true;
}

template <size_t L>
stat_t evaluate(const std::vector<Piece<L>> &lib, board_t<L> board, bool abort_on_zero) {
    using namespace std::placeholders;
    auto func = std::bind(solve_count<L>, lib, _1);
    boost::basic_thread_pool pool;
    std::vector<boost::future<size_t>> futures;
    [&](this auto &&self, auto it, Shape<L> curr) {
        if (it == board.regions.end()) {
            futures.emplace_back(boost::async(pool, func, curr));
            return;
        }
        for (auto pos : *it++)
            self(it, curr.clear(pos));
    }(board.regions.begin(), board.base);
    stat_t stat{};
    stat.min = std::numeric_limits<size_t>::max();
    auto total = 0zu;
    for (auto &f : futures) {
        f.wait();
        auto v = f.get();
        if (!v && abort_on_zero) {
            pool.interrupt_and_join();
            return { 1 };
        }
        if (v == 0) stat.zeros++;
        if (v == 1) stat.singles++;
        stat.min = std::min(stat.min, v);
        stat.max = std::max(stat.max, v);
        total += v;
    }
    stat.avg = static_cast<double>(total) / board.count;
    return stat;
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

int enumerate_library(size_t area, bool restrict_C, auto &&screening, char *argv[]) {
    auto min_piece_size = std::stoul(argv[1]);
    auto max_piece_size = std::stoul(argv[2]);
    auto min_pieces = std::stoul(argv[3]);
    auto max_pieces = std::stoul(argv[4]);
    auto fork_point = argv[5] ? std::stoul(argv[5]) : restrict_C ? 5 : 8;

    std::atomic<size_t> all_shapes{}, possible_shapes{};
    auto valid = ::fcntl(3, F_GETFD) != -1 || errno != EBADF;
    if (valid) {
        std::print("writing to fd:3 ({})\n",
            std::filesystem::read_symlink("/proc/self/fd/3").c_str());
    }
    auto a_threshold = 10zu;
    auto p_threshold = 10zu;
    boost::basic_thread_pool pool;
    std::atomic<size_t> running_tasks{}, pending_tasks{ 1zu };
    using shapes_t = std::vector<Shape<8>>;
    // one task, one *shapes
    [&](this auto &&self, size_t n, size_t i, size_t pieces, size_t left, std::shared_ptr<shapes_t> shapes, size_t depth) -> void {
        if (!depth)
            running_tasks.fetch_add(1, std::memory_order_relaxed);
        if (!left) {
            if (pieces < min_pieces)
                goto fin;
            if (auto a = ++all_shapes; a % a_threshold == 0) {
                std::print("all shapes = {} ...\n", a);
                a_threshold *= 10;
            }
            if (!screening(*shapes))
                goto fin;
            if (auto p = ++possible_shapes; p % p_threshold == 0) {
                std::print("possible shapes = {} ...\n", p);
                p_threshold *= 10;
            }
            if (valid) {
                auto sz = shapes->size();
                ::write(3, &sz, sizeof(sz));
                ::write(3, shapes->data(), shapes->size() * sizeof(Shape<8>));
            }
            goto fin;
        }
        while (n <= max_piece_size && i >= shape_count(n))
            n++, i = 0;
        if (n > max_piece_size || left < n || pieces >= max_pieces)
            goto fin;
        {
            auto fork = [&](Shape<8> sh) {
                if (depth < fork_point && running_tasks.load(std::memory_order_relaxed) >= 64) {
                    shapes->push_back(sh);
                    self(n, i + 1, pieces + 1, left - n, shapes, depth + 1); // taking
                    shapes->pop_back();
                } else {
                    auto shapes_next = std::make_shared<shapes_t>(*shapes); // copy
                    shapes_next->push_back(sh);
                    pending_tasks.fetch_add(1, std::memory_order_acquire);
                    boost::async(pool, self, n, i + 1, pieces + 1, left - n, shapes_next, 0); // taking
                }
            };
            fork(shape_at<8>(n, i));
            if (!restrict_C) {
                if (auto sh_flip = flip(shape_at<8>(n, i)); sh_flip)
                    fork(*sh_flip);
            }
            self(n, i + 1, pieces, left, shapes, depth + 1); // not taking
        }
    fin:
        if (!depth) {
            running_tasks.fetch_sub(1, std::memory_order_relaxed);
            if (pending_tasks.fetch_sub(1, std::memory_order_acq_rel) == 1)
                pool.close();
        }
    }(min_piece_size, 0, 0, area, std::make_shared<shapes_t>(), 0);
    pool.join();

    std::print("possible shapes = {}/{}\n", possible_shapes.load(), all_shapes.load());
    return 0;
}

#define L 8

int evaluate_libraries(const board_t<L> &board, char *argv[]) {
    (void)argv;
    while (true) {
        size_t sz;
        if (::read(3, &sz, sizeof(sz)) != sizeof(sz))
            break;
        std::vector<Piece<L>> pieces;
        pieces.reserve(sz);
        for (auto i = 0zu; i < sz; i++) {
            Shape<8> sh{ 0u };
            ::read(3, &sh, sizeof(sh));
            pieces.emplace_back(sh.to<L>());
        }
        auto stat = evaluate(pieces, board, true);
        if (!stat.zeros)
            std::print("zeros={} singles={} min={} max={} avg={}\n",
                    stat.zeros, stat.singles, stat.min, stat.max, stat.avg);
    }
    return 0;
}

int main(int argc, char *argv[]) {
    auto board = construct_board<L>(R"(
mmmmmm
mmmmmm
ddddddd
ddddddd
ddddddd
ddddddd
ddd
)");

    board = construct_board<L>(R"(
mmmmmmXX
mmmmmmXX
ddddddXX
ddddddXX
ddddddXX
ddddddXX
ddddddd
wwwwwww
)");

    std::print("working on a board of size={} left={} count={}\n",
            board.base.size(), board.base.size() - board.regions.size(), board.count);

    using namespace std::string_view_literals;

    if (argv[1] == "enumerate"sv)
        return enumerate_library(board.base.size() - board.regions.size(), false, [&](const auto &shapes) {
            return screen(shapes
                | std::views::transform([](Shape<8> sh){ return Piece<L>{ sh.to<L>() }; })
                | std::ranges::to<std::vector>(), board);
        }, ++argv);

    if (argv[1] == "enumerate-C"sv)
        return enumerate_library(board.base.size() - board.regions.size(), true, [&](const auto &shapes) {
            return screen(shapes
                | std::views::transform([](Shape<8> sh){ return Piece<L>{ sh.to<L>(), 0b01101001u }; })
                | std::ranges::to<std::vector>(), board);
        }, ++argv);

    if (argv[1] == "evaluate"sv)
        return evaluate_libraries(board, ++argv);
}
