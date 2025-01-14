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
    register_vector<Piece::Placement>("VPlacement");
    register_vector<Piece>("VPiece");
    enum_<SymmetryGroup>("SymmetryGroup")
        .value("C1", SymmetryGroup::C1)
        .value("C2", SymmetryGroup::C2)
        .value("C4", SymmetryGroup::C4)
        .value("D1_X", SymmetryGroup::D1_X)
        .value("D1_Y", SymmetryGroup::D1_Y)
        .value("D1_P", SymmetryGroup::D1_P)
        .value("D1_S", SymmetryGroup::D1_S)
        .value("D2_XY", SymmetryGroup::D2_XY)
        .value("D2_PS", SymmetryGroup::D2_PS)
        .value("D4", SymmetryGroup::D4)
        ;
    function("groupProduct", select_overload<SymmetryGroup(SymmetryGroup, SymmetryGroup)>(&::operator*));
    function("subgroup", select_overload<bool(SymmetryGroup, SymmetryGroup)>(&::operator>=));
    function("solve", &::solve, return_value_policy::take_ownership());
    class_<Piece::Placement>("Placement")
        .property("normal", &Piece::Placement::normal)
        .property("enabled", &Piece::Placement::enabled)
        .property("duplicate", &Piece::Placement::duplicate)
        ;
    class_<Piece>("Piece")
        .constructor<Shape>()
        .property("count", &Piece::count)
        .property("shape", &Piece::canonical)
        .property("placements", &Piece::placements, return_value_policy::reference())
        ;
    class_<Shape>("Shape")
        .constructor<Shape::shape_t>()
        .property("LEN", &Shape::get_LEN)
        .property("value", &Shape::get_value)
        .function("normalize", &Shape::normalize)
        .function("canonical_form", &Shape::canonical_form)
        .property("classify", &Shape::classify)
        .property("symmetry", &Shape::symmetry)
        .function("transform0", &Shape::transform<false, false, false>)
        .function("transform1", &Shape::transform<false, true,  false>)
        .function("transform2", &Shape::transform<false, false, true >)
        .function("transform3", &Shape::transform<false, true,  true >)
        .function("transform4", &Shape::transform<true,  false, false>)
        .function("transform5", &Shape::transform<true,  true,  false>)
        .function("transform6", &Shape::transform<true,  false, true >)
        .function("transform7", &Shape::transform<true,  true,  true >)
        .property("top", &Shape::top)
        .property("left", &Shape::left)
        .property("width", &Shape::width)
        .property("height", &Shape::height)
        .property("bottom", &Shape::bottom)
        .property("right", &Shape::right)
        .function("test", &Shape::test)
        .function("set", &Shape::set)
        .function("clear", &Shape::clear)
        ;
    class_<Step>("Step")
        .property("piece_id", &Step::piece_id)
        .property("trs_id", &Step::trs_id)
        .property("a", &Step::a)
        .property("b", &Step::b)
        .property("c", &Step::c)
        .property("d", &Step::d)
        .property("x", &Step::x)
        .property("y", &Step::y)
        .property("shape", &Step::shape)
        ;
    class_<Solution>("Solution")
        .property("steps", &Solution::steps, return_value_policy::reference())
        .property("map", &Solution::map, return_value_policy::reference())
        ;
}
