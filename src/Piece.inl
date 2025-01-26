#include "Piece.hpp"

#include <ranges>

bool Piece::cover(coords_t pos, auto &&func) const {
    auto [tgtY, tgtX] = pos;
    for (auto &&[p, trs] : std::views::zip(placements, std::views::iota(0zu))) {
        if (!p.enabled)
            continue;
        auto [maxY, maxX] = p.max;
        for (auto [bitY, bitX] : p.normal) {
            if (bitX > tgtX || bitX + maxX < tgtX)
                continue;
            if (bitY > tgtY || bitY + maxY < tgtY)
                continue;
            auto x = tgtX - bitX;
            auto y = tgtY - bitY;
            if (func(p.normal.translate_unsafe(x, y), trs, coords_t{ y, x }))
                return true;
        }
    }
    return false;
}

bool Piece::cover(auto &&func) const {
    for (auto &&[p, trs] : std::views::zip(placements, std::views::iota(0zu))) {
        if (!p.enabled)
            continue;
        auto [maxY, maxX] = p.max;
        for (auto x = 0; x <= maxX; x++)
            for (auto y = 0; y <= maxY; y++)
                if (func(p.normal.translate_unsafe(x, y), trs, coords_t{ y, x }))
                    return true;
    }
    return false;
}
