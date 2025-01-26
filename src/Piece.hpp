#pragma once

#include "Shape.hpp"

#include <initializer_list>
#include <vector>

using ssize_t = std::make_signed_t<size_t>;

template <size_t L>
struct Piece {
    size_t count;

    Shape<L> canonical;

    struct Placement {
        Shape<L> normal;
        coords_t max;
        bool enabled, duplicate;
    };

    std::vector<Placement> placements;

    friend class Library;

    Piece(Shape<L> s);
    bool cover(coords_t pos, auto &&func) const;
    bool cover(auto &&func) const;
};

template <size_t L>
struct Step {
    size_t piece_id, trs_id;
    int a, b, c, d, x, y;
    Shape<L> shape;
};

template <size_t L>
struct Solution {
    std::vector<Step<L>> steps;
    std::vector<std::vector<ssize_t>> map;

    explicit Solution(std::vector<Step<L>> st);
    Solution(const Solution &other) = default;
    Solution(Solution &&other) noexcept = default;
    Solution &operator=(const Solution &other) = default;
    Solution &operator=(Solution &&other) noexcept = default;
};

template <size_t L>
std::vector<Solution<L>> solve(const std::vector<Piece<L>> &lib, Shape<L> board, bool single);

template <size_t L>
size_t solve_count(const std::vector<Piece<L>> &lib, Shape<L> board);
