#include "frow.h"

#include <algorithm>
#include <compare>
#include <limits>
#include <optional>
#include <ranges>
#include <set>

#include "util.hpp"
#include "searcher.hpp"

std::optional<Board<8>> g_board;
std::optional<Naming> g_nme;
unsigned g_sym;

extern int n_devices;

struct tt_t {
    uint64_t shape;
    uint8_t nm;
};

static std::vector<frow_t> frowL[16], frowR[16];
frow_info_t h_frowInfoL[16], h_frowInfoR[16];

std::strong_ordering operator<=>(const frow_t &l, const frow_t &r) {
    if (l.shape < r.shape)
        return std::strong_ordering::less;
    if (l.shape > r.shape)
        return std::strong_ordering::greater;
    if (l.nm0123 < r.nm0123)
        return std::strong_ordering::less;
    if (l.nm0123 > r.nm0123)
        return std::strong_ordering::greater;
    return std::strong_ordering::equal;
}

void compute_frow_on_cpu(bool show_report) {
    using nm_t = decltype(tt_t::nm);
    auto count = 0zu;
    std::array<std::vector<tt_t>, 64> fanout;
    auto push_translate = [&](uint64_t nm, Shape<8> t) {
        for (auto row = 0u; row <= t.bottom(); row++)
            for (auto col = 0u; col <= t.right(); col++) {
                auto v = t.translate_unsafe(col, row).get_value();
                if (nm >= std::numeric_limits<nm_t>::max())
                    throw std::runtime_error{ "nm_t too small" };
                fanout[std::countr_zero(v)].emplace_back(v, nm);
            }
    };
    for (auto m = g_nme->min_m; m <= g_nme->max_m; m++) {
        for (auto i = 0zu; i < g_nme->arr_sizes[m]; i++) {
            count++;
            auto sh = Shape<8>(g_nme->arr[m][i]);
            for (auto s = g_sym; auto t : sh.transforms(true)) {
                if (s & 1u)
                    push_translate(g_nme->name_piece(m, i), t);
                s >>= 1;
            }
        }
    }
    for (auto pos = 0u; pos < 64u; pos++) {
        auto &f = fanout[pos];
        std::ranges::sort(f, std::less{}, &tt_t::shape);
        auto [end, _] = std::ranges::unique(f, std::ranges::equal_to{}, &tt_t::shape);
        f.erase(end, f.end());
    }

    char used[256]{};
    std::vector<tt_t> used_v;
    auto invest = [&](auto &&self, uint64_t empty_area, uint64_t mask, uint64_t original, auto &&obj) {
        if (!(empty_area & mask)) {
            std::vector<nm_t> nms(4, 0xff);
            for (auto &&[nm, tt] : std::views::zip(nms, used_v))
                nm = tt.nm;
            std::ranges::sort(nms);
            auto island_size = Shape<8>{ empty_area }.sml_island().size();
            size_t min_m;
            switch (g_nme->min_m) {
                case 1: // 0 => 1-piece; 1,2 => 2-pieces
                    if (nms[0] != 0)
                        min_m = 1;
                    else if (nms[1] != 1)
                        min_m = 2;
                    else
                        min_m = 3;
                    break;
                case 2: // 0,1 => 2-pieces
                    if (nms[1] != 1)
                        min_m = 2;
                    else
                        min_m = 3;
                    break;
                default:
                    min_m = g_nme->min_m;
                    break;
            }
            if (island_size && island_size < min_m)
                return;
            frow_t fr{ original & ~empty_area };
            fr.nm[0] = nms[0];
            fr.nm[1] = nms[1];
            fr.nm[2] = nms[2];
            fr.nm[3] = nms[3];
            obj.emplace(fr);
            return;
        }
        auto pos = std::countr_zero(empty_area);
        for (auto [shape, nm] : fanout[pos]) {
            if (used[nm]) [[unlikely]]
                continue;
            if (shape & ~empty_area) [[unlikely]]
                continue;
            used[nm] = 1, used_v.push_back({ shape, nm });
            self(self, empty_area & ~shape, mask, original, obj);
            used[nm] = 0, used_v.pop_back();
        }
    };
    size_t total_sz[6]{};
    auto regularize = [&](std::vector<frow_t> &f, const std::set<frow_t> &fs) {
        frow_info_t fi;
        f.reserve(fs.size());
        for (auto ff : fs)
            f.emplace_back(ff);
        fi.data = f.data();
        fi.data32 = new frow32_t[f.size()];
        fi.dataL = new uint32_t[f.size()];
        fi.dataH = new uint32_t[f.size()];
        fi.data0123 = new uint32_t[f.size()];
        for (auto i = 0zu; i < f.size(); i++) {
            fi.data32[i] = fi.data[i];
            fi.dataL[i] = fi.data32[i].shapeL;
            fi.dataH[i] = fi.data32[i].shapeH;
            fi.data0123[i] = fi.data32[i].nm0123;
        }
        for (auto i = 0; i < 6; i++) {
            fi.sz[i] = std::ranges::upper_bound(f,
                    (1ull << (8 * i + 8)) - 1ull, std::less{}, &frow_t::shape) - f.begin();
            total_sz[i] += fi.sz[i];
            if (show_report)
                std::print("/{}", fi.sz[i]);
        }
        if (fi.sz[5] != f.size())
            throw std::runtime_error{ "internal error" };
        return fi;
    };
    for (auto ea = 0u; ea < 16u; ea++) {
        std::set<frow_t> f;
        if (show_report)
            std::print("[0b1111{:04b}] => L", ea);
        invest(invest, ea | ~0b00001111ull, 0b00001111ull, ea | ~0b00001111ull, f);
        h_frowInfoL[ea] = regularize(frowL[ea], f);
        if (show_report)
            std::print("\n");
    }
    for (auto ea = 0u; ea < 16u; ea++) {
        std::set<frow_t> f;
        if (show_report)
            std::print("[0b{:04b}0000] => R", ea);
        invest(invest, (ea << 4) | ~0b11111111ull, 0b11110000ull, (ea << 4) | ~0b11111111ull, f);
        h_frowInfoR[ea] = regularize(frowR[ea], f);
        if (show_report)
            std::print("\n");
    }
}
