#pragma once

#include "Shape.hpp"

extern size_t shape_count(size_t n);
extern size_t shape_count_C(size_t n);

template <size_t L>
Shape<L> shape_at(size_t n, size_t i);
template <>
Shape<8> shape_at<8>(size_t n, size_t i);

extern template Shape<8> shape_at<8>(size_t n, size_t i);
extern template Shape<11> shape_at<11>(size_t n, size_t i);
