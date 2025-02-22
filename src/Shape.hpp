#pragma once

#include <array>
#include <bit>
#include <compare>
#include <functional>
#include <string_view>
#include <vector>

#include "Group.hpp"

using coords_t = std::pair<int, int>; // Y, X

template <size_t L>
class Shape {
public:
    static constexpr size_t LEN = L;
    [[nodiscard]] auto get_LEN() const { return LEN; }

    // [LSB] [1] [2] [3]
    // [4] ...
    // [8] ...
    // [12] ...    [MSB]
    using shape_t = std::conditional_t<LEN * LEN <= 64, uint64_t, __uint128_t>;

private:
    static constexpr size_t BITS = sizeof(shape_t) * 8;

    static constexpr shape_t FULL = LEN * LEN == BITS
        ? ~shape_t{ 0u }
        : (shape_t{ 1u } << (LEN * LEN % BITS)) - 1u;
    static constexpr shape_t FIRST_ROW = (shape_t{ 1 } << LEN) - 1u;
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

    explicit constexpr Shape(std::string_view sv) : value{} {
        if (sv.empty()) return;
        if (sv.front() == '\n')
            sv = sv.substr(1);
        auto row_mask = shape_t{ 1u };
        auto mask = row_mask;
        for (auto ch : sv) {
            switch (ch) {
                case ' ':
                case '0':
                case '.':
                case '-':
                case '_':
                    mask <<= 1u;
                    break;
                case '*':
                case '1':
                case '@':
                case '#':
                case 'X':
                case '%':
                    value |= mask;
                    mask <<= 1u;
                    break;
                case '\n':
                    mask = row_mask <<= LEN;
                    break;
            }
        }
    }

    template <size_t M>
    explicit constexpr Shape(const Shape<M> &other) : value{} {
        if constexpr (L == M) {
            value = other.value;
        } else {
            auto sh = Shape{ 0u };
            for (auto [y, x] : other)
                sh = sh.set(y, x);
            *this = sh;
        }
    }

    SymmetryGroup classify() const {
        return static_cast<SymmetryGroup>(symmetry());
    }

    constexpr Shape(const Shape &v) = default;
    constexpr Shape(Shape &&v) noexcept = default;
    constexpr Shape &operator=(const Shape &v) noexcept = default;
    constexpr Shape &operator=(Shape &&v) noexcept = default;

    constexpr operator bool() const { return value; }
    [[nodiscard]] constexpr bool operator==(const Shape &other) const {
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
        if constexpr (std::is_same_v<shape_t, __uint128_t>) {
            return std::popcount(static_cast<uint64_t>(value))
                + std::popcount(static_cast<uint64_t>(value >> 64u));
        } else {
            return std::popcount(value);
        }
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
    [[nodiscard]] Shape test(coords_t pos) const {
        return test(pos.first, pos.second);
    }

    [[nodiscard]] Shape set(size_t row, size_t col) const {
        return Shape{ value | shape_t{ 1u } << (row * LEN + col) };
    }
    [[nodiscard]] Shape set(coords_t pos) const {
        return set(pos.first, pos.second);
    }

    [[nodiscard]] Shape clear(size_t row, size_t col) const {
        return Shape{ value & ~(shape_t{ 1u } << (row * LEN + col)) };
    }
    [[nodiscard]] Shape clear(coords_t pos) const {
        return clear(pos.first, pos.second);
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
    [[nodiscard]] Shape translate_unsafe(int x, int y) const;
    [[nodiscard]] Shape translate(coords_t d) const {
        return translate(d.second, d.first);
    }
    [[nodiscard]] Shape translate_unsafe(coords_t d) const {
        return translate_unsafe(d.second, d.first);
    }

    constexpr auto front() const {
        auto id = std::countr_zero(value);
        return std::make_pair(id / LEN, id % LEN);
    }

    constexpr auto back() const {
        auto id = std::bit_width(value) - 1u;
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

    [[nodiscard]] Shape sml_island() const;

    [[nodiscard]] std::string to_string() const;

    template <size_t M>
    [[nodiscard]] Shape<M> to() const {
        return Shape<M>(*this);
    }
};

constexpr inline Shape<8> operator ""_s8(const char *str, size_t len) {
    return Shape<8>(std::string_view{ str, len });
}
constexpr inline Shape<11> operator ""_s11(const char *str, size_t len) {
    return Shape<11>(std::string_view{ str, len });
}

#ifdef FMT_VERSION
template <size_t L>
struct fmt::formatter<Shape<L>> : formatter<string_view> {
    auto format(Shape<L> c, format_context &ctx) const
        -> format_context::iterator {
        return formatter<string_view>::format(c.to_string(), ctx);
    }
};
#endif

template <size_t L>
struct std::hash<Shape<L>> {
    constexpr size_t operator()(Shape<L> s) const noexcept {
        return s.value;
    }
};

extern template Shape<8> Shape<8>::transform<false, false, false>(bool norm) const;
extern template Shape<8> Shape<8>::transform<false, true,  false>(bool norm) const;
extern template Shape<8> Shape<8>::transform<false, false, true >(bool norm) const;
extern template Shape<8> Shape<8>::transform<false, true,  true >(bool norm) const;
extern template Shape<8> Shape<8>::transform<true,  false, false>(bool norm) const;
extern template Shape<8> Shape<8>::transform<true,  true,  false>(bool norm) const;
extern template Shape<8> Shape<8>::transform<true,  false, true >(bool norm) const;
extern template Shape<8> Shape<8>::transform<true,  true,  true >(bool norm) const;

extern template std::array<Shape<8>, 8> Shape<8>::transforms(bool norm) const;
extern template Shape<8> Shape<8>::translate(int x, int y) const;
extern template Shape<8> Shape<8>::translate_unsafe(int x, int y) const;
extern template Shape<8> Shape<8>::canonical_form(unsigned forms) const;
extern template unsigned Shape<8>::symmetry() const;
extern template Shape<8> Shape<8>::extend1() const;
extern template bool Shape<8>::connected() const;
extern template Shape<8> Shape<8>::sml_island() const;
extern template std::string Shape<8>::to_string() const;

extern template Shape<11> Shape<11>::transform<false, false, false>(bool norm) const;
extern template Shape<11> Shape<11>::transform<false, true,  false>(bool norm) const;
extern template Shape<11> Shape<11>::transform<false, false, true >(bool norm) const;
extern template Shape<11> Shape<11>::transform<false, true,  true >(bool norm) const;
extern template Shape<11> Shape<11>::transform<true,  false, false>(bool norm) const;
extern template Shape<11> Shape<11>::transform<true,  true,  false>(bool norm) const;
extern template Shape<11> Shape<11>::transform<true,  false, true >(bool norm) const;
extern template Shape<11> Shape<11>::transform<true,  true,  true >(bool norm) const;

extern template std::array<Shape<11>, 8> Shape<11>::transforms(bool norm) const;
extern template Shape<11> Shape<11>::translate(int x, int y) const;
extern template Shape<11> Shape<11>::translate_unsafe(int x, int y) const;
extern template Shape<11> Shape<11>::canonical_form(unsigned forms) const;
extern template unsigned Shape<11>::symmetry() const;
extern template Shape<11> Shape<11>::extend1() const;
extern template bool Shape<11>::connected() const;
extern template Shape<11> Shape<11>::sml_island() const;
extern template std::string Shape<11>::to_string() const;
