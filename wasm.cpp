#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <sys/types.h>
using namespace emscripten;

#include "Shape.hpp"
#include "Piece.hpp"

EMSCRIPTEN_BINDINGS(pieces) {
    class_<Library>("Library")
        .constructor<Library>()
        .function("push", &Library::push)
        ;
    class_<Shape>("Shape")
        .class_property("LEN", &Shape::LEN)
        .property("value", &Shape::get_value)
        ;
    class_<Step>("Step")
        .property("piece_id", &Step::piece_id)
        .property("shape", &Step::shape)
        ;
    register_vector<Step>("VectorSteps");
    class_<Solution>("Solution")
        .property("steps", &Solution::steps)
        .property("map", &Solution::map)
        ;
}
