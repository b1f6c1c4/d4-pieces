#pragma once

#include "Shape.hpp"

#include <algorithm>
#include <ranges>

class Piece {
    Shape canonical;

    struct Placement {
        Shape normal;
        coords_t max;
    };

    std::vector<Placement> placements;

public:
    Piece(Shape::shape_t sh) : canonical{ Shape{ sh }.canonical_form() } {
        for (auto sh : canonical.transforms(true)) {
            if (std::ranges::find(placements, sh, &Placement::normal) != placements.end())
                continue;

            placements.emplace_back(sh,
                    std::make_pair(Shape::LEN - sh.bottom(), Shape::LEN - sh.right()));
        }
    }

    void cover(coords_t pos, auto &&func) const {
        auto [tgtY, tgtX] = pos;
        for (auto &p : placements) {
            auto [maxY, maxX] = pos;
            for (auto [bitY, bitX] : p.normal.bits()) {
                if (bitX > tgtX || bitX + maxX < tgtX)
                    continue;
                if (bitY > tgtY || bitY + maxY < tgtY)
                    continue;
                auto placed = p.normal.translate(tgtX - bitX, tgtY - bitY);
                func(placed);
            }
        }
    }
};
