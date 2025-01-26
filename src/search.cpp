#include <algorithm>
#include <filesystem>
#include <cerrno>
#include <fcntl.h>
#include <ranges>
#include <print>
#include <list>

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
stat_t evaluate(const std::vector<Piece<L>> &lib, board_t<L> board) {
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
    auto total = 0zu;
    auto [min, max] = std::ranges::minmax(futures | std::views::transform([&](auto &f) {
        f.wait();
        auto v = f.get();
        if (v == 0) stat.zeros++;
        if (v == 1) stat.singles++;
        total += v;
        return v;
    }));
    stat.min = min;
    stat.max = max;
    stat.avg = static_cast<double>(total) / board.count;
    return stat;
}

int enumerate_library(size_t area, char *argv[]) {
    static constexpr size_t L = 8;

    auto min_piece_size = std::stoul(argv[1]);
    auto max_piece_size = std::stoul(argv[2]);
    auto min_pieces = std::stoul(argv[3]);
    auto max_pieces = std::stoul(argv[4]);

    std::vector<Shape<L>> shapes;
    size_t possible_shapes{};
    auto valid = ::fcntl(3, F_GETFD) != -1 || errno != EBADF;
    if (valid) {
        std::print("writing to fd:3 ({})\n",
            std::filesystem::read_symlink("/proc/self/fd/3").c_str());
    }
    [&](this auto &&self, size_t n, size_t i, size_t pieces, size_t left) {
        if (!left) {
            if (pieces >= min_pieces) {
                possible_shapes++;
                if (valid) {
                    auto sz = shapes.size();
                    ::write(3, &sz, sizeof(sz));
                    ::write(3, shapes.data(), shapes.size() * sizeof(Shape<L>));
                }
            }
            return;
        }
        while (n <= max_piece_size && i >= shape_count(n))
            n++, i = 0;
        if (n > max_piece_size || left < n || pieces >= max_pieces)
            return;
        auto sh = shape_at<L>(n, i++);
        self(n, i, pieces, left); // not taking
        shapes.push_back(sh);
        self(n, i, pieces + 1, left - n); // taking
        shapes.pop_back();
    }(min_piece_size, 0, 0, area);

    std::print("possible shapes = {}\n", possible_shapes);
    return 0;
}

#define L 8

int main(int argc, char *argv[]) {
    auto board = construct_board<L>(R"(
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
        return enumerate_library(board.base.size() - board.regions.size(), ++argv);

    // auto stat = evaluate(pieces, board);
    // std::print("zeros={} singles={} min={} max={} avg={}\n",
    //         stat.zeros, stat.singles, stat.min, stat.max, stat.avg);
}
