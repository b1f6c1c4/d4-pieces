#pragma once

#include <boost/integer.hpp>
#include <fmt/base.h>
#include <functional>
#include <bit>
#include <compare>
#include <array>
#include <vector>

using coords_t = std::pair<int, int>; // Y, X

enum class SymmetryGroup : uint16_t { C1, C2, C4, D1, D2, D4 };

template <>
struct fmt::formatter<SymmetryGroup> : formatter<string_view> {
    auto format(SymmetryGroup c, format_context &ctx) const
        -> format_context::iterator;
};

class Shape {
public:
    static constexpr size_t LEN = 8;
    // [LSB] [1] [2] [3]
    // [4] ...
    // [8] ...
    // [12] ...    [MSB]
    using shape_t = boost::uint_t<LEN * LEN>::least;

private:
    static constexpr size_t BITS = sizeof(shape_t) * 8;

    static constexpr size_t FULL = static_cast<shape_t>(
            LEN * LEN == BITS ? ~0ull : (1ull << (LEN * LEN % BITS)) - 1ull);
    static constexpr size_t FIRST_ROW = static_cast<shape_t>((1ull << LEN) - 1ull);
    static constexpr size_t FIRST_COL = [] constexpr {
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

    [[nodiscard]] constexpr Shape operator+(Shape other) const {
        return Shape{ value | ~other.value };
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

    [[nodiscard]] Shape clear(size_t row, size_t col) const {
        return Shape{ value & ~(1ull << (row * LEN + col)) };
    }

    // C1 C2 C4 D1 D2 D4
    // =  =  =  =  =  =  false, false, false => identity
    //          ?  ?  =  false, true,  false => flip X
    //          ?  ?  =  false, false, true  => flip Y
    //    =  =     =  =  false, true,  true  => rot180
    //          ?  ?  =  true,  false, false => flip primary
    //       =        =  true,  true,  false => rot90 CW
    //       =        =  true,  false, true  => rot90 CCW
    //          ?  ?  =  true,  true,  true  => flip secondary
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
    constexpr auto bits() const {
        std::vector<coords_t> pos;
        for (auto v = value; v; v -= v & -v) {
            auto id = std::countr_zero(v);
            pos.emplace_back(id / LEN, id % LEN);
        }
        return pos;
    }

    // aligned to top-left, rotated/flipped if possible
    [[nodiscard]] Shape canonical_form() const;

    [[nodiscard]] bool connected() const;
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

extern template Shape Shape::transform<false, false, false>(bool norm) const;
extern template Shape Shape::transform<false, true,  false>(bool norm) const;
extern template Shape Shape::transform<false, false, true >(bool norm) const;
extern template Shape Shape::transform<false, true,  true >(bool norm) const;
extern template Shape Shape::transform<true,  false, false>(bool norm) const;
extern template Shape Shape::transform<true,  true,  false>(bool norm) const;
extern template Shape Shape::transform<true,  false, true >(bool norm) const;
extern template Shape Shape::transform<true,  true,  true >(bool norm) const;
