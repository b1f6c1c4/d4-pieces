#include "Shape.hpp"
#include <string>
#include <fmt/format.h>
#include <ranges>

template <bool Swap, bool FlipX, bool FlipY>
Shape Shape::transform(bool norm) const {
    if (!value) return Shape{ 0u };
    shape_t out{};
    // iterate from out's MSB to LSB
    for (auto i = 0zu; i < LEN * LEN; i++) {
        auto out_row = (LEN * LEN - i - 1) / LEN;
        auto out_col = (LEN * LEN - i - 1) % LEN;
        out <<= 1;
        auto in_row = Swap ? out_col : out_row;
        auto in_col = Swap ? out_row : out_col;
        if constexpr (FlipX) in_col = LEN - in_col - 1;
        if constexpr (FlipY) in_row = LEN - in_row - 1;
        out |= test(in_row, in_col);
    }
    auto sh = Shape{ out };
    if (norm)
        return sh.normalize();
    return sh;
}

template Shape Shape::transform<false, false, false>(bool norm) const;
template Shape Shape::transform<false, true,  false>(bool norm) const;
template Shape Shape::transform<false, false, true >(bool norm) const;
template Shape Shape::transform<false, true,  true >(bool norm) const;
template Shape Shape::transform<true,  false, false>(bool norm) const;
template Shape Shape::transform<true,  true,  false>(bool norm) const;
template Shape Shape::transform<true,  false, true >(bool norm) const;
template Shape Shape::transform<true,  true,  true >(bool norm) const;

std::array<Shape, 8> Shape::transforms(bool norm) const {
    return {
        transform<false, false, false>(norm),
        transform<false, true,  false>(norm),
        transform<false, false, true >(norm),
        transform<false, true,  true >(norm),
        transform<true,  false, false>(norm),
        transform<true,  true,  false>(norm),
        transform<true,  false, true >(norm),
        transform<true,  true,  true >(norm),
    };
}

Shape Shape::translate(int x, int y) const {
    auto v = value;
    while (y > 0) v <<= LEN, y--;
    while (y < 0) v >>= LEN, y++;
    while (x > 0) v <<= 1u, v &= ~FIRST_COL, x--;
    while (x < 0) v &= ~FIRST_COL, v >>= 1u, x++;
    return Shape{ v };
}

Shape Shape::canonical_form() const {
    return std::ranges::min(transforms(true), std::less{}, &Shape::value);
}

bool Shape::connected() const {
    if (!value) return false;
    auto visited = static_cast<shape_t>(value & -value);
    if (visited == value) return true;
    while (true) {
        auto north = visited >> LEN;
        auto south = visited << LEN;
        auto west = (visited & ~FIRST_COL) >> 1u;
        auto east = (visited << 1u) & ~FIRST_COL;
        auto next = (visited | north | south | west | east) & value;
        if (next == value)
            return true;
        if (next == visited)
            return false;
        visited = next;
    }
}

SymmetryGroup Shape::classify() const {
    auto base = normalize();
    auto x = base == transform<false, true, false>(true);
    auto y = base == transform<false, false, true>(true);
    auto p = base == transform<true, false, false>(true);
    auto s = base == transform<true, true, true>(true);
    if (x && y && p && s)
        return SymmetryGroup::D4;
    if (x && y || p && s)
        return SymmetryGroup::D2;
    if (x || y || p || s)
        return SymmetryGroup::D1;
    if (*this == transform<true, true, false>(true))
        return SymmetryGroup::C4;
    if (*this == transform<false, true, true>(true))
        return SymmetryGroup::C2;
    return SymmetryGroup::C1;
}

auto fmt::formatter<SymmetryGroup>::format(SymmetryGroup c, format_context &ctx) const
    -> format_context::iterator {
    string_view name = "unknown";
    switch (c) {
        case SymmetryGroup::C1: name = "C1"; break;
        case SymmetryGroup::C2: name = "C2"; break;
        case SymmetryGroup::C4: name = "C4"; break;
        case SymmetryGroup::D1: name = "D1"; break;
        case SymmetryGroup::D2: name = "D2"; break;
        case SymmetryGroup::D4: name = "D4"; break;
    }
    return formatter<string_view>::format(name, ctx);
}

auto fmt::formatter<Shape>::format(Shape c, format_context &ctx) const
    -> format_context::iterator {
    std::string txt;
    for (auto row = 0u; row < Shape::LEN; row++) {
        for (auto col = 0u; col < Shape::LEN; col++)
            if (c.test(row, col))
                txt.push_back('@');
            else
                txt.push_back(' ');
        txt.push_back('\n');
    }
    return formatter<string_view>::format(txt, ctx);
}
