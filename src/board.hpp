#pragma once

#include <fstream>
#include <ranges>
#include <sstream>

#include "Shape.hpp"

template <size_t L>
struct Board {
    Shape<L> base;
    std::vector<Shape<L>> regions;
    size_t count;

    explicit constexpr Board(std::string_view sv) : base{ 0u } {
        std::unordered_map<char, std::string> map;
        auto valid = [](char ch) {
            return (ch >= '0' && ch <= '9')
                || (ch >= 'A' && ch <= 'Z')
                || (ch >= 'a' && ch <= 'z');
        };
        std::string the_base;
        for (auto ch : sv)
            if (valid(ch))
                map.emplace(ch, "");
        for (auto ch : sv) {
            the_base.push_back(valid(ch) ? '#' : ch);
            for (auto &[k, v] : map)
                v.push_back(valid(ch) ? k == ch ? '#' : '.' : ch);
        }
        base = Shape<L>{ the_base };
        for (auto sh : map
                | std::views::transform([](auto &kv){ return Shape<L>{ kv.second }; }))
            regions.push_back(sh);
        count = 1zu;
        for (auto r : regions)
            count *= r.size();
    }

    static inline Board<L> from_file(auto &&path) {
        std::stringstream buffer;
        std::ifstream fin(path);
        buffer << fin.rdbuf();
        return Board<L>(std::string_view(buffer.str()));
    }

    void foreach(auto &&func) {
        auto f = [&,end=regions.end()](auto &&self, auto it, Shape<L> curr) {
            if (it == end) {
                func(curr);
                return;
            }
            for (auto pos : *it++)
                self(self, it, curr.clear(pos));
        };
        f(f, regions.begin(), base);
    }
};

constexpr inline Board<8> operator ""_b8(const char *str, size_t len) {
    return Board<8>({ str, len });
}

constexpr inline Board<11> operator ""_b11(const char *str, size_t len) {
    return Board<11>({ str, len });
}
