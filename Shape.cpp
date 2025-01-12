#include "Shape.hpp"
#include <string>
#include <fmt/format.h>

template <bool Swap, bool FlipX, bool FlipY>
Shape Shape::transform() const {
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
    return Shape{ normalize(out) };
}

template Shape Shape::transform<false, false, false>() const;
template Shape Shape::transform<false, true,  false>() const;
template Shape Shape::transform<false, false, true >() const;
template Shape Shape::transform<false, true,  true >() const;
template Shape Shape::transform<true,  false, false>() const;
template Shape Shape::transform<true,  true,  false>() const;
template Shape Shape::transform<true,  false, true >() const;
template Shape Shape::transform<true,  true,  true >() const;

Shape Shape::canonical_form() const {
    return Shape{ std::min({
        transform<false, false, false>().value,
        transform<false, true,  false>().value,
        transform<false, false, true >().value,
        transform<false, true,  true >().value,
        transform<true,  false, false>().value,
        transform<true,  true,  false>().value,
        transform<true,  false, true >().value,
        transform<true,  true,  true >().value,
    }) };
}

bool Shape::connected() const {
    if (!value) return false;
    auto visited = static_cast<shape_t>(value & -value);
    if (visited == value) return true;
    while (true) {
        auto north = visited >> 4u;
        auto south = visited << 4u;
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
    auto x = *this == transform<false, true, false>();
    auto y = *this == transform<false, false, true>();
    auto p = *this == transform<true, true, false>();
    auto s = *this == transform<true, true, true>();
    if (x && y && p && s)
        return SymmetryGroup::D4;
    if (x && y || p && s)
        return SymmetryGroup::D2;
    if (x || y || p || s)
        return SymmetryGroup::D1;
    if (*this == transform<false, true, true>())
        return SymmetryGroup::C2;
    return SymmetryGroup::C1;
}

auto fmt::formatter<SymmetryGroup>::format(SymmetryGroup c, format_context &ctx) const
    -> format_context::iterator {
    string_view name = "unknown";
    switch (c) {
        case SymmetryGroup::C1: name = "C1"; break;
        case SymmetryGroup::C2: name = "C2"; break;
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
