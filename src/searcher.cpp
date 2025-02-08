#include "searcher.hpp"
#include "searcher_cuda.h"

#include <iostream>
#include <print>
#include <chrono>

#include "naming.inl"

#define BOOST_THREAD_VERSION 5
#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/thread/future.hpp>
#include <boost/container/small_vector.hpp>

uint64_t Searcher::step(Shape<8> empty_area) {
    std::print("{}CudaSearcher::CudaSearcher(ea={})\n", empty_area.to_string(), empty_area.size());
    CudaSearcher cs{ empty_area.get_value() };
    while (cs.get_height()) {
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
    std::terminate();

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
