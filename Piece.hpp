#pragma once

#include "Shape.hpp"

#include <initializer_list>
#include <vector>

struct Piece {
    size_t count;

    Shape canonical;

    struct Placement {
        Shape normal;
        coords_t max;
        bool enabled, duplicate;
    };

    std::vector<Placement> placements;

    friend class Library;

    Piece(Shape s);
    void cover(coords_t pos, auto &&func) const;
};

struct Step {
    size_t piece_id;
    Shape shape;
};

struct Solution {
    std::vector<Step> steps;
    std::vector<std::vector<ssize_t>> map;

    explicit Solution(std::vector<Step> st);
    Solution(const Solution &other) = default;
    Solution(Solution &&other) noexcept = default;
    Solution &operator=(const Solution &other) = default;
    Solution &operator=(Solution &&other) noexcept = default;
};

std::vector<Solution> solve(const std::vector<Piece> &lib, Shape board);
