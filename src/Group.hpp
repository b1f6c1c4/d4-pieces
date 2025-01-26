#pragma once

#include <bit>
#include <cstdint>
#include <cstddef>
#include <type_traits>

enum class SymmetryGroup : uint16_t {
  C1    = 0b00000001u,
  C2    = 0b00001001u,
  C4    = 0b01101001u,
  D1_X  = 0b00000011u,
  D1_Y  = 0b00000101u,
  D1_P  = 0b00010001u,
  D1_S  = 0b10000001u,
  D2_XY = 0b00001111u,
  D2_PS = 0b10011001u,
  D4    = 0b11111111u,
};

constexpr inline bool operator>=(SymmetryGroup lhs, SymmetryGroup rhs) {
    auto l = static_cast<std::underlying_type<SymmetryGroup>::type>(lhs);
    auto r = static_cast<std::underlying_type<SymmetryGroup>::type>(rhs);
    return (l & r) == r;
}

constexpr inline SymmetryGroup operator*(SymmetryGroup lhs, SymmetryGroup rhs) {
    if (lhs >= rhs) return lhs;
    if (rhs >= lhs) return rhs;
    auto l = static_cast<std::underlying_type<SymmetryGroup>::type>(lhs);
    auto r = static_cast<std::underlying_type<SymmetryGroup>::type>(rhs);
    auto lor = l | r;
    auto test = [&](unsigned v, unsigned x) { if ((lor & v) == v) lor |= x; };
    test(0b00000111u, 0b00001000u); // flip XY == rot180
    test(0b10010001u, 0b00001000u); // flip PS == rot180
    test(0b00001011u, 0b00000100u); // rot180 + flip X == flip Y
    test(0b00001101u, 0b00000010u); // rot180 + flip Y == flip X
    test(0b00011001u, 0b10000000u); // rot180 + flip P == flip S
    test(0b10001001u, 0b00010000u); // rot180 + flip S == flip P
    test(0b00100011u, 0b11111110u); // rot90 + flip X == D4
    test(0b01000011u, 0b11111110u); // rot90 + flip X == D4
    test(0b00100101u, 0b11111110u); // rot90 + flip X == D4
    test(0b01000101u, 0b11111110u); // rot90 + flip X == D4
    test(0b00010011u, 0b11111110u); // flip XP == D4
    test(0b10000011u, 0b11111110u); // flip XS == D4
    test(0b00010101u, 0b11111110u); // flip YP == D4
    test(0b10000101u, 0b11111110u); // flip YS == D4
    return static_cast<SymmetryGroup>(lor);
}

constexpr inline size_t order(SymmetryGroup v) {
    auto s = static_cast<std::underlying_type<SymmetryGroup>::type>(v);
    return 8 / std::popcount(s);
}

#ifdef FMT_VERSION
template <>
struct fmt::formatter<SymmetryGroup> : formatter<string_view> {
    auto format(SymmetryGroup c, format_context &ctx) const
        -> format_context::iterator;
};
#endif
