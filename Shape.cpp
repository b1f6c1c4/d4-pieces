#include "Shape.hpp"
#include <string>
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

Shape Shape::canonical_form(unsigned forms) const {
    auto v = normalize().value;
    for (auto sh : transforms(true)) {
        if (forms & 1u)
            v = std::min(v, sh.value);
        forms >>= 1u;
    }
    return Shape{ v };
}

unsigned Shape::symmetry() const {
    auto c = normalize();
    auto s = 0u;
    auto m = 1u;
    for (auto sh : transforms(true)) {
        if (c == sh)
            s |= m;
        m <<= 1u;
    }
    return s;
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
    return static_cast<SymmetryGroup>(symmetry());
}

static_assert(SymmetryGroup::D4 >= SymmetryGroup::D2_XY);
static_assert(SymmetryGroup::D4 >= SymmetryGroup::D2_PS);
static_assert(SymmetryGroup::D2_XY >= SymmetryGroup::D1_X);
static_assert(SymmetryGroup::D2_XY >= SymmetryGroup::D1_Y);
static_assert(SymmetryGroup::D2_PS >= SymmetryGroup::D1_P);
static_assert(SymmetryGroup::D2_PS >= SymmetryGroup::D1_S);

static_assert(SymmetryGroup::D4 >= SymmetryGroup::C4);
static_assert(SymmetryGroup::D2_XY >= SymmetryGroup::C2);
static_assert(SymmetryGroup::D2_PS >= SymmetryGroup::C2);
static_assert(SymmetryGroup::D1_X >= SymmetryGroup::C1);
static_assert(SymmetryGroup::D1_Y >= SymmetryGroup::C1);
static_assert(SymmetryGroup::D1_P >= SymmetryGroup::C1);
static_assert(SymmetryGroup::D1_S >= SymmetryGroup::C1);

static_assert(SymmetryGroup::C4 >= SymmetryGroup::C2);
static_assert(SymmetryGroup::C2 >= SymmetryGroup::C1);

static_assert(SymmetryGroup::D4 * SymmetryGroup::D4 == SymmetryGroup::D4);
static_assert(SymmetryGroup::D2_XY * SymmetryGroup::D2_XY == SymmetryGroup::D2_XY);
static_assert(SymmetryGroup::D2_PS * SymmetryGroup::D2_PS == SymmetryGroup::D2_PS);
static_assert(SymmetryGroup::D1_X * SymmetryGroup::D1_X == SymmetryGroup::D1_X);
static_assert(SymmetryGroup::D1_Y * SymmetryGroup::D1_Y == SymmetryGroup::D1_Y);
static_assert(SymmetryGroup::D1_P * SymmetryGroup::D1_P == SymmetryGroup::D1_P);
static_assert(SymmetryGroup::D1_S * SymmetryGroup::D1_S == SymmetryGroup::D1_S);
static_assert(SymmetryGroup::C4 * SymmetryGroup::C4 == SymmetryGroup::C4);
static_assert(SymmetryGroup::C2 * SymmetryGroup::C2 == SymmetryGroup::C2);
static_assert(SymmetryGroup::C1 * SymmetryGroup::C1 == SymmetryGroup::C1);

static_assert(SymmetryGroup::C4 * SymmetryGroup::D2_XY == SymmetryGroup::D4);

#ifdef FMT_VERSION
auto fmt::formatter<SymmetryGroup>::format(SymmetryGroup c, format_context &ctx) const
    -> format_context::iterator {
    string_view name = "unknown";
    switch (c) {
        case SymmetryGroup::C1: name = "C1"; break;
        case SymmetryGroup::C2: name = "C2"; break;
        case SymmetryGroup::C4: name = "C4"; break;
        case SymmetryGroup::D1: name = "D1"; break;
        case SymmetryGroup::D1_X: name = "D1_X"; break;
        case SymmetryGroup::D1_Y: name = "D1_Y"; break;
        case SymmetryGroup::D1_P: name = "D1_P"; break;
        case SymmetryGroup::D1_S: name = "D1_S"; break;
        case SymmetryGroup::D2: name = "D2"; break;
        case SymmetryGroup::D2_XY: name = "D2_XY"; break;
        case SymmetryGroup::D2_PS: name = "D2_PS"; break;
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
#endif
