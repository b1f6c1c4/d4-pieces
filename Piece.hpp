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

    Solution(std::vector<Step> history);
};

std::vector<Solution> solve(const std::vector<Piece> &lib, Shape board);
