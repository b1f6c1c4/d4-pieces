#include <algorithm>
#include <ranges>
#include <print>

#define BOOST_THREAD_VERSION 5
#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/thread/future.hpp>

#include "Piece.hpp"
#include "Shape.hpp"
#include "Piece.inl"

using namespace std::placeholders;

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

int main() {
    std::vector<Shape<8>::shape_t> sh{
        0b11110000'10000000u,          // L
        0b11100000'00110000u,          // Z
        0b01100000'01000000'11000000u, // S
        0b10000000'10000000'11100000u, // L_
        0b11100000'11100000u,          // rect
        0b11100000'11000000u,          // rect-
        0b11100000'10100000u,          // C
        0b11110000'00100000u,          // |-
    };
    auto pieces = sh
        | std::views::transform([](Shape<8>::shape_t sh){
              return Piece<11>{ Shape<11>{ Shape<8>{ sh } } };
          })
        | std::ranges::to<std::vector>();
    //auto board = R"(
//mmmmmmXX
//mmmmmmXX
//ddddddXX
//ddddddXX
//ddddddXX
//ddddddXX
//ddddddd
//wwwwwww
//)"_b8;
    auto board = R"(
mmmmmm
mmmmmm
ddddddd
ddddddd
ddddddd
ddddddd
ddd
)"_b11;

    std::print("count={}\n", board.count);

    auto stat = evaluate(pieces, board);
    std::print("zeros={} singles={} min={} max={} avg={}\n",
            stat.zeros, stat.singles, stat.min, stat.max, stat.avg);
}
