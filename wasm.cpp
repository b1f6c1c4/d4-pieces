#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <sys/types.h>
using namespace emscripten;

#include "Shape.hpp"
#include "Piece.hpp"

EMSCRIPTEN_BINDINGS(pieces) {
    register_vector<Step>("VStep");
    register_vector<ssize_t>("VSSize");
    register_vector<std::vector<ssize_t>>("VVSSize");
    register_vector<Solution>("VSolution");
    class_<Library>("Library")
        .constructor()
        .function("push", &Library::push)
        .property("length", &Library::size)
        .function("at", &Library::at, return_value_policy::reference())
        .function("solve", &Library::solve, return_value_policy::take_ownership())
        ;
    class_<Piece>("Piece")
        .property("count", &Piece::count)
        .property("shape", &Piece::shape)
        ;
    class_<Shape>("Shape")
        .constructor<Shape::shape_t>()
        .property("LEN", &Shape::get_LEN)
        .property("value", &Shape::get_value)
        .property("width", &Shape::width)
        .property("height", &Shape::height)
        .function("test", &Shape::test)
        .function("set", &Shape::set)
        .function("clear", &Shape::clear)
        ;
    class_<Step>("Step")
        .property("piece_id", &Step::piece_id)
        .property("shape", &Step::shape)
        ;
    class_<Solution>("Solution")
        .property("steps", &Solution::steps)
        .property("map", &Solution::map)
        ;
}
