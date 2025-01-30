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
    fcf_cache(g_nme->size_pieces());
    std::print("moved to GPU\n");
}

uint64_t Searcher::step(Shape<8> empty_area) {
    auto cnt = 0ull;
    // find all shapes covering the first empty block while also fits
    CudaSearcher cs{ g_nme->size_pieces() };
    cs.start_search(empty_area.get_value());
    for (const unsigned char *ptr; (ptr = cs.next());) {
      std::print("{{ ");
      for (auto i = 0; i < 8; i++)
        std::print("0x{:08x}, ", *(uint32_t*)&ptr[4 * i]);
      std::print("}}\n");
    }
    std::abort();
    for (const unsigned char *ptr; (ptr = cs.next());) {
        char arr[256]{};
        for (auto i = 0; i < 32; i++)
            if (arr[ptr[i]]++)
                throw std::runtime_error{ "CUDA gives duplicated pieces" };
        auto id = g_nme->name([&](uint64_t m, uint64_t i){
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
