#pragma once

#include <boost/integer.hpp>
#include <fmt/base.h>
#include <functional>
#include <algorithm>
#include <bit>

enum class SymmetryGroup { C1, C2, D1, D2, D4 };

template <>
struct fmt::formatter<SymmetryGroup> : formatter<string_view> {
    auto format(SymmetryGroup c, format_context &ctx) const
        -> format_context::iterator;
};

class Shape {
public:
    static constexpr size_t LEN = 4;
private:
    // [LSB] [1] [2] [3]
    // [4] ...
    // [8] ...
    // [12] ...    [MSB]
    using shape_t = boost::uint_t<LEN * LEN>::least;

    static constexpr size_t FULL = static_cast<shape_t>((1ull << (LEN * LEN)) - 1u);
    static constexpr size_t FIRST_ROW = static_cast<shape_t>((1ull << LEN) - 1u);
    static constexpr size_t FIRST_COL = [] constexpr {
        shape_t total{};
        shape_t mask{ 1 };
        for (auto i = 0zu; i < LEN; i++)
            total |= mask, mask <<= LEN;
        return total;
    }();

    shape_t value;

    friend std::hash<Shape>;

    static constexpr shape_t normalize(shape_t sh) {
        while (!(sh & FIRST_ROW))
            sh >>= 4u;
        while (!(sh & FIRST_COL))
            sh >>= 1u;
        return sh;
    }

public:
    explicit constexpr Shape(shape_t v) : value{ shape_t(v & FULL) } { }
    constexpr Shape(const Shape &v) = default;
    constexpr Shape(Shape &&v) noexcept = default;
    constexpr Shape &operator=(const Shape &v) noexcept = default;
    constexpr Shape &operator=(Shape &&v) noexcept = default;
    constexpr bool operator==(const Shape &other) const = default;

    constexpr size_t size() const {
        return std::popcount(value);
    }

    bool test(size_t row, size_t col) const {
        return (value >> (row * LEN + col)) & 1u;
    }

    // false, false, false => identity
    // false, true,  false => flipX
    // false, false, true  => flipY
    // false, true,  true  => rot180
    // true,  false, false => flip primary
    // true,  true,  false => rot90 CW
    // true,  false, true  => rot90 CCW
    // true,  true,  true  => flip secondary
    template <bool Swap, bool FlipX, bool FlipY>
    Shape transform() const;

    Shape canonical_form() const;

    bool connected() const;

    SymmetryGroup classify() const;
};

template <>
struct fmt::formatter<Shape> : formatter<string_view> {
    auto format(Shape c, format_context &ctx) const
        -> format_context::iterator;
};

template <>
struct std::hash<Shape> {
    constexpr size_t operator()(Shape s) const noexcept {
        return s.value;
    }
};

extern template Shape Shape::transform<false, false, false>() const;
extern template Shape Shape::transform<false, true,  false>() const;
extern template Shape Shape::transform<false, false, true >() const;
extern template Shape Shape::transform<false, true,  true >() const;
extern template Shape Shape::transform<true,  false, false>() const;
extern template Shape Shape::transform<true,  true,  false>() const;
extern template Shape Shape::transform<true,  false, true >() const;
extern template Shape Shape::transform<true,  true,  true >() const;
