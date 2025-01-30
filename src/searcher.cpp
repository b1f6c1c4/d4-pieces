#include "searcher.hpp"
#include "searcher_cuda.h"

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

tt_t *fast_canonical_form;
size_t fast_canonical_forms;

void compute_fast_canonical_form() {
    using nm_t = decltype(tt_t::nm);
    std::unordered_map<uint64_t, nm_t> map;
    auto count = 0zu;
    auto push_translate = [&](uint64_t nm, Shape<8> t) {
        for (auto row = 0u; row <= t.bottom(); row++)
            for (auto col = 0u; col <= t.right(); col++) {
                auto v = t.translate_unsafe(col, row).get_value();
                if (nm >= std::numeric_limits<nm_t>::max())
                    throw std::runtime_error{ "nm_t too small" };
                map.emplace(v, nm).second;
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
    fast_canonical_form = new tt_t[map.size()]; // uninitialized
    fast_canonical_forms = map.size();
    for (auto ptr = fast_canonical_form; auto [k, v] : map)
        *ptr++ = tt_t{ k, v };
    std::print("cached {} => {} canonical forms\n", fast_canonical_forms, count);
    fcf_cache();
    std::print("moved to GPU\n");
}

uint64_t Searcher::step(Shape<8> empty_area) {
    if (!empty_area) {
        if (n_used_pieces < g_nme->min_n || n_used_pieces > g_nme->max_n)
            return 0;
        auto id = g_nme->name([this](uint64_t m, uint64_t i){
            return !!used_pieces[g_nme->name_piece(m, i)];
        });
        if (id) {
            return log(*id);
        } else {
            std::print("Warning: Naming rejected SearcherCRTP's plan\n");
            return 0;
        }
    }
    if (n_used_pieces == g_nme->max_n)
        return 0;

    auto cnt = 0ull;
    // find all shapes covering the first empty block while also fits
    searcher_step(empty_area.get_value(), nullptr);
    return cnt;
}

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
