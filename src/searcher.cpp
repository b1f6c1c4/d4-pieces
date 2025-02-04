#include "searcher.hpp"
#include "searcher_cuda.h"

#include <array>
#include <bit>
#include <compare>
#include <set>
#include <iostream>
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

    char used[256]{};
    std::vector<tt_t> used_v;
    auto invest = [&](this auto &&self, uint64_t empty_area, uint64_t mask, uint64_t original, auto &&obj) {
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
            self(empty_area & ~shape, mask, original, obj);
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
        for (auto i = 0; i < 6; i++) {
            fi.sz[i] = std::ranges::upper_bound(f,
                    (1ull << (8 * i + 8)) - 1ull, std::less{}, &frow_t::shape) - f.begin();
            total_sz[i] += fi.sz[i];
            std::print("/{}", fi.sz[i]);
        }
        if (fi.sz[5] != f.size())
            throw std::runtime_error{ "internal error" };
        return fi;
    };
    for (auto ea = 0u; ea < 16u; ea++) {
        std::set<frow_t> f;
        std::print("[0b1111{:04b}] => L", ea);
        invest(ea | ~0b00001111ull, 0b00001111ull, ea | ~0b00001111ull, f);
        frowInfoL[ea] = regularize(frowL[ea], f);
        std::print("\n");
    }
    for (auto ea = 0u; ea < 16u; ea++) {
        std::set<frow_t> f;
        std::print("[0b{:04b}0000] => R", ea);
        invest((ea << 4) | ~0b11111111ull, 0b11110000ull, (ea << 4) | ~0b11111111ull, f);
        frowInfoR[ea] = regularize(frowR[ea], f);
        std::print("\n");
    }
    for (auto i = 0; i < 6; i++)
        std::print("sz[{}] = {} = {}KiB\n", i, total_sz[i],
                total_sz[i] * sizeof(frow_t) / 1024.0);
    frow_cache(frowInfoL.data(), frowInfoR.data());
}

uint64_t Searcher::step(Shape<8> empty_area) {
    std::print("{}CudaSearcher::CudaSearcher(ea={})\n", empty_area.to_string(), empty_area.size());
    CudaSearcher cs{ empty_area.get_value() };
    while (true) {
        std::print("height={} size={}={:.02f}GiB next={}={:.02f}GiB avg={:.1f}x \n",
                cs.get_height(),
                cs.size(), sizeof(R) * cs.size() / 1073741824.0,
                cs.next_size(), sizeof(R) * cs.next_size() / 1073741824.0,
                1.0 * cs.next_size() / cs.size());
        auto t1 = std::chrono::steady_clock::now();
        cs.search_GPU();
        auto t2 = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        if (us < 1000)
            std::print("  => completed in {}us\n", us);
        else if (us < 1000000)
            std::print("  => completed in {:.2f}ms\n", us / 1e3);
        else
            std::print("  => completed in {:.2f}s\n", us / 1e6);
    }

    /*
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
