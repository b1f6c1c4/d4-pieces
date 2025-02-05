#pragma once

#include <cstddef>
#include "record.h"

struct CudaSearcher;
namespace boost::executors { class basic_thread_pool; }
struct CSR;
struct Sorter {
    explicit Sorter(CudaSearcher &p);
    ~Sorter();

    void push(Rg<RX> r, unsigned height);
    void join();

private:
    CudaSearcher &parent;
    size_t dedup, total;
    boost::executors::basic_thread_pool *pool;
    CSR *sets;
};
