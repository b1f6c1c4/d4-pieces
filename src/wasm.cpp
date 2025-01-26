#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <sys/types.h>
using namespace emscripten;

#include "Shape.hpp"
#include "known.hpp"
#include "Piece.hpp"

EMSCRIPTEN_BINDINGS(pieces) {
    register_vector<ssize_t>("VSSize");
    register_vector<std::vector<ssize_t>>("VVSSize");
    register_vector<Step<8>>("VStep8");
    register_vector<Piece<8>>("VPiece8");
    register_vector<Piece<8>::Placement>("VPlacement8");
    register_vector<Solution<8>>("VSolution8");
    register_vector<Step<11>>("VStep11");
    register_vector<Piece<11>>("VPiece11");
    register_vector<Piece<11>::Placement>("VPlacement11");
    register_vector<Solution<11>>("VSolution11");
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
    function("solve8", &::solve<8>, return_value_policy::take_ownership());
    function("solve11", &::solve<11>, return_value_policy::take_ownership());
    class_<Piece<8>::Placement>("Placement8")
        .property("normal", &Piece<8>::Placement::normal)
        .property("enabled", &Piece<8>::Placement::enabled)
        .property("duplicate", &Piece<8>::Placement::duplicate)
        ;
    class_<Piece<8>>("Piece8")
        .constructor<Shape<8>>()
        .property("count", &Piece<8>::count)
        .property("shape", &Piece<8>::canonical)
        .property("placements", &Piece<8>::placements, return_value_policy::reference())
        ;
    class_<Shape<8>>("Shape8")
        .constructor<Shape<8>::shape_t>()
        .property("LEN", &Shape<8>::get_LEN)
        .property("value", &Shape<8>::get_value)
        .property("size", &Shape<8>::size)
        .function("normalize", &Shape<8>::normalize)
        .function("canonical_form", &Shape<8>::canonical_form)
        .property("classify", &Shape<8>::classify)
        .property("symmetry", &Shape<8>::symmetry)
        .function("transform0", &Shape<8>::transform<false, false, false>)
        .function("transform1", &Shape<8>::transform<false, true,  false>)
        .function("transform2", &Shape<8>::transform<false, false, true >)
        .function("transform3", &Shape<8>::transform<false, true,  true >)
        .function("transform4", &Shape<8>::transform<true,  false, false>)
        .function("transform5", &Shape<8>::transform<true,  true,  false>)
        .function("transform6", &Shape<8>::transform<true,  false, true >)
        .function("transform7", &Shape<8>::transform<true,  true,  true >)
        .property("top", &Shape<8>::top)
        .property("left", &Shape<8>::left)
        .property("width", &Shape<8>::width)
        .property("height", &Shape<8>::height)
        .property("bottom", &Shape<8>::bottom)
        .property("right", &Shape<8>::right)
        .function("test", &Shape<8>::test)
        .function("set", &Shape<8>::set)
        .function("clear", &Shape<8>::clear)
        ;
    class_<Step<8>>("Step8")
        .property("piece_id", &Step<8>::piece_id)
        .property("trs_id", &Step<8>::trs_id)
        .property("a", &Step<8>::a)
        .property("b", &Step<8>::b)
        .property("c", &Step<8>::c)
        .property("d", &Step<8>::d)
        .property("x", &Step<8>::x)
        .property("y", &Step<8>::y)
        .property("shape", &Step<8>::shape)
        ;
    class_<Solution<8>>("Solution")
        .property("steps", &Solution<8>::steps, return_value_policy::reference())
        .property("map", &Solution<8>::map, return_value_policy::reference())
        ;
    class_<Piece<11>::Placement>("Placement11")
        .property("normal", &Piece<11>::Placement::normal)
        .property("enabled", &Piece<11>::Placement::enabled)
        .property("duplicate", &Piece<11>::Placement::duplicate)
        ;
    class_<Piece<11>>("Piece11")
        .constructor<Shape<11>>()
        .property("count", &Piece<11>::count)
        .property("shape", &Piece<11>::canonical)
        .property("placements", &Piece<11>::placements, return_value_policy::reference())
        ;
    class_<Shape<11>>("Shape11")
        .constructor<Shape<11>::shape_t>()
        .property("LEN", &Shape<11>::get_LEN)
        .property("value", &Shape<11>::get_value)
        .property("size", &Shape<11>::size)
        .function("normalize", &Shape<11>::normalize)
        .function("canonical_form", &Shape<11>::canonical_form)
        .property("classify", &Shape<11>::classify)
        .property("symmetry", &Shape<11>::symmetry)
        .function("transform0", &Shape<11>::transform<false, false, false>)
        .function("transform1", &Shape<11>::transform<false, true,  false>)
        .function("transform2", &Shape<11>::transform<false, false, true >)
        .function("transform3", &Shape<11>::transform<false, true,  true >)
        .function("transform4", &Shape<11>::transform<true,  false, false>)
        .function("transform5", &Shape<11>::transform<true,  true,  false>)
        .function("transform6", &Shape<11>::transform<true,  false, true >)
        .function("transform7", &Shape<11>::transform<true,  true,  true >)
        .property("top", &Shape<11>::top)
        .property("left", &Shape<11>::left)
        .property("width", &Shape<11>::width)
        .property("height", &Shape<11>::height)
        .property("bottom", &Shape<11>::bottom)
        .property("right", &Shape<11>::right)
        .function("test", &Shape<11>::test)
        .function("set", &Shape<11>::set)
        .function("clear", &Shape<11>::clear)
        ;
    class_<Step<11>>("Step11")
        .property("piece_id", &Step<11>::piece_id)
        .property("trs_id", &Step<11>::trs_id)
        .property("a", &Step<11>::a)
        .property("b", &Step<11>::b)
        .property("c", &Step<11>::c)
        .property("d", &Step<11>::d)
        .property("x", &Step<11>::x)
        .property("y", &Step<11>::y)
        .property("shape", &Step<11>::shape)
        ;
    class_<Solution<11>>("Solution")
        .property("steps", &Solution<11>::steps, return_value_policy::reference())
        .property("map", &Solution<11>::map, return_value_policy::reference())
        ;
    function("shape_count", &::shape_count);
    function("shape_at8", &::shape_at<8>);
    function("shape_at11", &::shape_at<11>);
}
