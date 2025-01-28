#pragma once

#include <cstdint>
#include <functional>
#include <bit>
#include <compare>
#include <type_traits>
#include <array>
#include <vector>

using coords_t = std::pair<int, int>; // Y, X

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

class Shape {
public:
    static constexpr size_t LEN = 8;
    [[nodiscard]] auto get_LEN() const { return LEN; }

    // [LSB] [1] [2] [3]
    // [4] ...
    // [8] ...
    // [12] ...    [MSB]
    using shape_t = uint64_t;

private:
    static constexpr size_t BITS = sizeof(shape_t) * 8;

    static constexpr shape_t FULL = static_cast<shape_t>(
            LEN * LEN == BITS ? ~0ull : (1ull << (LEN * LEN % BITS)) - 1ull);
    static constexpr shape_t FIRST_ROW = static_cast<shape_t>((1ull << LEN) - 1ull);
    static constexpr shape_t FIRST_COL = [] constexpr {
        shape_t total{};
        shape_t mask{ 1 };
        for (auto i = 0zu; i < LEN; i++)
            total |= mask, mask <<= LEN;
        return total;
    }();

    shape_t value;

    friend std::hash<Shape>;

public:
    explicit constexpr Shape(shape_t v)
        : value{ shape_t(v & FULL) } { }

    SymmetryGroup classify() const;

    constexpr Shape(const Shape &v) = default;
    constexpr Shape(Shape &&v) noexcept = default;
    constexpr Shape &operator=(const Shape &v) noexcept = default;
    constexpr Shape &operator=(Shape &&v) noexcept = default;

    constexpr operator bool() const { return value; }
    [[nodiscard]] constexpr auto operator==(const Shape &other) const {
        return value == other.value;
    }
    [[nodiscard]] constexpr auto operator<=>(const Shape &other) const {
        if (value == other.value)
            return std::partial_ordering::equivalent;
        if ((value & other.value) == value)
            return std::partial_ordering::less;
        if ((value & other.value) == other.value)
            return std::partial_ordering::greater;
        return std::partial_ordering::unordered;
    }

    [[nodiscard]] constexpr shape_t get_value() const { return value; }

    [[nodiscard]] constexpr size_t size() const {
        return std::popcount(value);
    }

    [[nodiscard]] constexpr size_t left() const {
        auto w = 0;
        auto v = value;
        for (; !(v & FIRST_COL); v >>= 1u)
            w++;
        return w;
    }

    // excluding left margin
    [[nodiscard]] constexpr size_t width() const {
        auto w = 0;
        auto v = value;
        for (; !(v & FIRST_COL); v >>= 1u);
        for (; v & FIRST_COL; v >>= 1u)
            w++;
        return w;
    }

    [[nodiscard]] constexpr size_t right() const {
        return LEN - left() - width();
    }

    [[nodiscard]] constexpr size_t top() const {
        auto h = 0;
        auto v = value;
        for (; !(v & FIRST_ROW); v >>= LEN)
            h++;
        return h;
    }

    // excluding top margin
    [[nodiscard]] constexpr size_t height() const {
        auto h = 0;
        auto v = value;
        for (; !(v & FIRST_ROW); v >>= LEN);
        for (; v & FIRST_ROW; v >>= LEN)
            h++;
        return h;
    }

    [[nodiscard]] constexpr size_t bottom() const {
        return LEN - top() - height();
    }

    [[nodiscard]] constexpr Shape operator|(Shape other) const {
        return Shape{ value | other.value };
    }

    [[nodiscard]] constexpr Shape operator&(Shape other) const {
        return Shape{ value & other.value };
    }

    [[nodiscard]] constexpr Shape operator-(Shape other) const {
        return Shape{ value & ~other.value };
    }

    [[nodiscard]] constexpr Shape normalize() const {
        if (!value)
            return Shape{ value };
        auto v = value;
        while (!(v & FIRST_ROW))
            v >>= LEN;
        while (!(v & FIRST_COL))
            v >>= 1u;
        return Shape{ v };
    }

    [[nodiscard]] bool test(size_t row, size_t col) const {
        return (value >> (row * LEN + col)) & 1u;
    }

    [[nodiscard]] Shape set(size_t row, size_t col) const {
        return Shape{ value | 1ull << (row * LEN + col) };
    }

    [[nodiscard]] Shape set(coords_t pos) const {
        return set(pos.first, pos.second);
    }

    [[nodiscard]] Shape clear(size_t row, size_t col) const {
        return Shape{ value & ~(1ull << (row * LEN + col)) };
    }

    // C1 C2 C4 D1 D2 D4
    // =  =  =  =  =  =  false, false, false =>   identity
    //          ?  ?  =  false, true,  false => * flip X
    //          ?  ?  =  false, false, true  => * flip Y
    //    =  =     =  =  false, true,  true  =>   rot180
    //          ?  ?  =  true,  false, false => * flip primary
    //       =        =  true,  true,  false =>   rot90 CW
    //       =        =  true,  false, true  =>   rot90 CCW
    //          ?  ?  =  true,  true,  true  => * flip secondary
    template <bool Swap, bool FlipX, bool FlipY>
    [[nodiscard]] Shape transform(bool norm) const;

    [[nodiscard]] std::array<Shape, 8> transforms(bool norm) const;

    //   -y
    // -x  +x
    //   +y
    [[nodiscard]] Shape translate(int x, int y) const;
    [[nodiscard]] Shape translate(coords_t d) const {
        return translate(d.second, d.first);
    }

    constexpr auto front() const {
        auto id = std::countr_zero(value);
        return std::make_pair(id / LEN, id % LEN);
    }

    struct bits_proxy {
        shape_t v;
        bool operator==(const bits_proxy &other) const = default;
        constexpr coords_t operator*() {
            auto id = std::countr_zero(v);
            return { id / LEN, id % LEN };
        }
        constexpr bits_proxy &operator++() {
            v -= v & -v;
            return *this;
        }
    };
    constexpr bits_proxy begin() const {
        return { value };
    }
    constexpr bits_proxy end() const {
        return { 0 };
    }

    // aligned to top-left, rotated/flipped if possible
    // LSB = identity
    // MSB = flip secondary
    [[nodiscard]] Shape canonical_form(unsigned forms = 0b11111111u) const;

    [[nodiscard]] unsigned symmetry() const;

    [[nodiscard]] Shape front_shape() const {
        return Shape{ static_cast<shape_t>(value & -value) };
    }

    [[nodiscard]] Shape extend1() const;

    [[nodiscard]] bool connected() const;
};

#ifdef FMT_VERSION
template <>
struct fmt::formatter<SymmetryGroup> : formatter<string_view> {
    auto format(SymmetryGroup c, format_context &ctx) const
        -> format_context::iterator;
};
template <>
struct fmt::formatter<Shape> : formatter<string_view> {
    auto format(Shape c, format_context &ctx) const
        -> format_context::iterator;
};
#endif

template <>
struct std::hash<Shape> {
    constexpr size_t operator()(Shape s) const noexcept {
        return s.value;
    }
};

extern template Shape Shape::transform<false, false, false>(bool norm) const;
extern template Shape Shape::transform<false, true,  false>(bool norm) const;
extern template Shape Shape::transform<false, false, true >(bool norm) const;
extern template Shape Shape::transform<false, true,  true >(bool norm) const;
extern template Shape Shape::transform<true,  false, false>(bool norm) const;
extern template Shape Shape::transform<true,  true,  false>(bool norm) const;
extern template Shape Shape::transform<true,  false, true >(bool norm) const;
extern template Shape Shape::transform<true,  true,  true >(bool norm) const;
