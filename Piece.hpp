#pragma once

#include "Shape.hpp"

#include <initializer_list>
#include <vector>

struct Piece {
    size_t count;

private:
    Shape canonical;

    struct Placement {
        Shape normal;
        coords_t max;
    };

    std::vector<Placement> placements;

    friend class Library;

public:
    Piece(Shape s);
    [[nodiscard]] Shape shape() const { return canonical; }
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

    void push(Shape sh);
    [[nodiscard]] auto size() const { return lib.size(); }
    [[nodiscard]] auto &at(size_t i) { return lib[i]; }

    std::vector<Solution> solve(Shape board) const;
};
