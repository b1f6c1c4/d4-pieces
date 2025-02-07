#pragma once

#include <cstddef>
#include <deque>
#include "record.h"
#include "region.h"

struct CudaSearcher;
namespace boost::executors { class basic_thread_pool; }
struct CSR;
struct Sorter {
    Sorter();
    ~Sorter();

    void push(Rg<RX> r);
    [[nodiscard]] std::deque<WL> join();

    unsigned print_stats() const;

private:
    size_t dedup, total, pending;
    boost::executors::basic_thread_pool *pool;
    CSR *sets;
};
