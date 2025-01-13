#pragma once

#include "Shape.hpp"

#include <initializer_list>
#include <vector>

struct Piece {
    Shape canonical;

    struct Placement {
        Shape normal;
        coords_t max;
    };

    std::vector<Placement> placements;
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

class Library {
    std::vector<Piece> lib;

public:
    Library() { }
    Library(std::initializer_list<Shape::shape_t> lst);

    bool push(Shape sh);

    std::vector<Solution> solve(Shape board) const;
};
