#include "Group.hpp"

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
#endif
