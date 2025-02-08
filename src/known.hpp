#pragma once

#ifndef BMARK
#include "Shape.hpp"
#endif
#include <cstdint>
#include <cstddef>

extern const uint64_t * const known_shapes[];
extern const size_t shapes_count[];
extern const uint64_t * const known_C_shapes[];
extern const size_t shapes_C_count[];

#ifndef BMARK
template <size_t L>
Shape<L> shape_at(size_t n, size_t i) {
    return Shape<8>{ known_shapes[n][i] }.to<L>();
}
#endif
