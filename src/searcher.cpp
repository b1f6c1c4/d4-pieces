#include "searcher.hpp"
#include "searcher_cuda.h"

#include <array>
#include <bit>
#include <compare>
#include <set>
#include <print>
#include <chrono>

#include "naming.inl"

#define BOOST_THREAD_VERSION 5
#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/thread/future.hpp>
#include <boost/container/small_vector.hpp>

std::optional<Board<8>> g_board;
std::optional<Naming> g_nme;
unsigned g_sym;

void show_devices() { }

static std::array<std::vector<frow_t>, 16> frowL, frowR;
static std::array<frow_info_t, 16> frowInfoL, frowInfoR;

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

void compute_fast_canonical_form() {
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

    for (auto ea = 1u; ea < 16u; ea++) {
        char used[256]{};
        std::vector<tt_t> used_v;
        std::set<frow_t> fL, fR;
        auto invest = [&](this auto &&self, uint64_t empty_area, uint64_t mask, uint64_t original, auto &&obj) {
            if (!(empty_area & mask)) {
                std::vector<nm_t> nms(4, 0xff);
                for (auto &&[nm, tt] : std::views::zip(nms, used_v))
                    nm = tt.nm;
                std::ranges::sort(nms);
                obj.emplace(frow_t{
                        (original & ~empty_area),
                        ((uint32_t)nms[3] << 24) |
                        ((uint32_t)nms[2] << 16) |
                        ((uint32_t)nms[1] <<  8) |
                        ((uint32_t)nms[0] <<  0) });
                return;
            }
            auto pos = std::countr_zero(empty_area);
            for (auto [shape, nm] : fanout[pos]) {
                if (used[nm]) [[unlikely]]
                    continue;
                if (shape & ~empty_area) [[unlikely]]
                    continue;
                used[nm] = 1, used_v.push_back({ shape, nm });
                self(empty_area & ~shape, mask, original, obj);
                used[nm] = 0, used_v.pop_back();
            }
        };
        auto regularize = [&](std::vector<frow_t> &f, const std::set<frow_t> &fs) {
            frow_info_t fi;
            f.reserve(fs.size());
            for (auto ff : fs)
                f.emplace_back(ff);
            fi.data = f.data();
            for (auto i = 0; i < 5; i++) {
                fi.sz[i] = std::ranges::upper_bound(f,
                        0b11111111ull << (8 * i), std::less{}, &frow_t::shape) - f.begin();
                std::print("/{}", fi.sz[i]);
            }
            if (fi.sz[4] != f.size())
                throw std::runtime_error{ "internal error" };
            return fi;
        };
        std::print("[0b{:08b}] => L", ea);
        invest(ea | ~0b00001111u, 0b00001111u, ea | ~0b00001111u, fL);
        frowInfoL[ea] = regularize(frowL[ea], fL);
        std::print(" R");
        invest((ea << 4) | ~0b11111111u, 0b11110000u, (ea << 4) | ~0b11111111u, fR);
        frowInfoR[ea] = regularize(frowL[ea], fR);
        std::print("\n");
    }
    std::abort();
}

uint64_t Searcher::step(Shape<8> empty_area) { /*
    auto cnt = 0ull;
    // find all shapes covering the first empty block while also fits
    CudaSearcher cs{ g_nme->size_pieces() };
    cs.start_search(empty_area.get_value());
    for (const unsigned char *ptr; (ptr = cs.next());) {
        std::print("{{ ");
        for (auto i = 0; i < 7; i++)
            std::print("0x{:08x}, ", *(uint32_t*)&ptr[4 * i]);
        std::print("}}\n");
        char arr[256]{};
        for (auto i = 0; i < 28; i++)
            if (ptr[i] != 255)
                if (arr[ptr[i]]++)
                    std::print("CUDA gives duplicated pieces {}", +ptr[i]);
                    // throw std::runtime_error{ std::format("CUDA gives duplicated pieces {}", +ptr[i]) };
        auto id = g_nme->name([&](uint64_t m, uint64_t i) {
            return !!arr[g_nme->name_piece(m, i)];
        });
        if (id) {
            if (log(*id))
                cnt++;
        } else {
            std::print("Warning: Naming rejected SearcherCRTP's plan\n");
        }
    }
    return cnt;
    */ return 0;
}

void SearcherFactory::run1() {
    g_board->foreach([&,i=0](Shape<8> sh) mutable {
        if (should_run(i, sh)) {
                auto *obj = make();
                obj->config_index = i;
                auto t1 = std::chrono::steady_clock::now();
                auto cnt = obj->step(sh);
                auto t2 = std::chrono::steady_clock::now();
                delete obj;
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
                after_run(i, sh, cnt, ms);
                configs_counter.fetch_add(1, std::memory_order_relaxed);
            configs_issue_counter++;
        }
        i++;
    });
};

void SearcherFactory::run() {
    boost::basic_thread_pool pool;
    g_board->foreach([&,i=0](Shape<8> sh) mutable {
        if (should_run(i, sh)) {
            boost::async(pool, [&,i,sh] {
                auto *obj = make();
                obj->config_index = i;
                auto t1 = std::chrono::steady_clock::now();
                auto cnt = obj->step(sh);
                auto t2 = std::chrono::steady_clock::now();
                delete obj;
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
                after_run(i, sh, cnt, ms);
                configs_counter.fetch_add(1, std::memory_order_relaxed);
            });
            configs_issue_counter++;
        }
        i++;
    });
    pool.close();
    pool.join();
};

void SearcherFactory::after_run(uint64_t i, Shape<8> sh, uint64_t cnt, uint64_t ms) {
    if (!cnt) {
        std::print("########### ERROR: a board with ZERO cnt found\n{}#######################\n",
                sh.to_string());
    }
}
