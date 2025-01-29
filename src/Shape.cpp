#include "Shape.hpp"
#include <string>
#include <ranges>

template <size_t L>
template <bool Swap, bool FlipX, bool FlipY>
Shape<L> Shape<L>::transform(bool norm) const {
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

template <size_t L>
std::array<Shape<L>, 8> Shape<L>::transforms(bool norm) const {
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

template <size_t L>
Shape<L> Shape<L>::translate(int x, int y) const {
    auto v = value;
    while (y > 0) v <<= LEN, y--;
    while (y < 0) v >>= LEN, y++;
    while (x > 0) v <<= 1u, v &= ~FIRST_COL, x--;
    while (x < 0) v &= ~FIRST_COL, v >>= 1u, x++;
    return Shape{ v };
}

template <size_t L>
Shape<L> Shape<L>::translate_unsafe(int x, int y) const {
    return Shape{ value << LEN * y + x };
}

template <size_t L>
Shape<L> Shape<L>::canonical_form(unsigned forms) const {
    auto v = ~shape_t{};
    for (auto sh : transforms(true)) {
        if (forms & 1u)
            v = std::min(v, sh.value);
        forms >>= 1u;
    }
    return Shape<L>{ v };
}

template <size_t L>
unsigned Shape<L>::symmetry() const {
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

template <size_t L>
Shape<L> Shape<L>::extend1() const {
    auto north = value >> LEN;
    auto south = value << LEN;
    auto west = (value & ~FIRST_COL) >> 1u;
    auto east = (value << 1u) & ~FIRST_COL;
    return Shape{ value | north | south | west | east };
}

template <size_t L>
bool Shape<L>::connected() const {
    if (!value) return false;
    auto visited = front_shape();
    if (visited == value) return true;
    while (true) {
        auto next = Shape{ visited }.extend1() & *this;
        if (next == *this)
            return true;
        if (next == visited)
            return false;
        visited = next;
    }
}

template <size_t L>
std::string Shape<L>::to_string() const {
    std::string txt;
    for (auto row = 0u; row < Shape<L>::LEN; row++) {
        for (auto col = 0u; col < Shape<L>::LEN; col++)
            if (test(row, col))
                txt.push_back('@');
            else
                txt.push_back(' ');
        txt.push_back('\n');
    }
    return txt;
}

template Shape<8> Shape<8>::transform<false, false, false>(bool norm) const;
template Shape<8> Shape<8>::transform<false, true,  false>(bool norm) const;
template Shape<8> Shape<8>::transform<false, false, true >(bool norm) const;
template Shape<8> Shape<8>::transform<false, true,  true >(bool norm) const;
template Shape<8> Shape<8>::transform<true,  false, false>(bool norm) const;
template Shape<8> Shape<8>::transform<true,  true,  false>(bool norm) const;
template Shape<8> Shape<8>::transform<true,  false, true >(bool norm) const;
template Shape<8> Shape<8>::transform<true,  true,  true >(bool norm) const;

template std::array<Shape<8>, 8> Shape<8>::transforms(bool norm) const;
template Shape<8> Shape<8>::translate(int x, int y) const;
template Shape<8> Shape<8>::translate_unsafe(int x, int y) const;
template Shape<8> Shape<8>::canonical_form(unsigned forms) const;
template unsigned Shape<8>::symmetry() const;
template Shape<8> Shape<8>::extend1() const;
template bool Shape<8>::connected() const;
template std::string Shape<8>::to_string() const;

template Shape<11> Shape<11>::transform<false, false, false>(bool norm) const;
template Shape<11> Shape<11>::transform<false, true,  false>(bool norm) const;
template Shape<11> Shape<11>::transform<false, false, true >(bool norm) const;
template Shape<11> Shape<11>::transform<false, true,  true >(bool norm) const;
template Shape<11> Shape<11>::transform<true,  false, false>(bool norm) const;
template Shape<11> Shape<11>::transform<true,  true,  false>(bool norm) const;
template Shape<11> Shape<11>::transform<true,  false, true >(bool norm) const;
template Shape<11> Shape<11>::transform<true,  true,  true >(bool norm) const;

template std::array<Shape<11>, 8> Shape<11>::transforms(bool norm) const;
template Shape<11> Shape<11>::translate(int x, int y) const;
template Shape<11> Shape<11>::translate_unsafe(int x, int y) const;
template Shape<11> Shape<11>::canonical_form(unsigned forms) const;
template unsigned Shape<11>::symmetry() const;
template Shape<11> Shape<11>::extend1() const;
template bool Shape<11>::connected() const;
template std::string Shape<11>::to_string() const;
