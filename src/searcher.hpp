#pragma once

#include <atomic>
#include <optional>
#include <vector>

#include "board.hpp"
#include "naming.hpp"

// defined in search.cpp
extern std::optional<Board<8>> g_board;
extern std::optional<Naming> g_nme;
extern unsigned g_sym;

// NOT thread-safe at all!
// DO NOT REUSE
class Searcher {
    bool shortcut;
    size_t n_used_pieces;
    std::vector<char> used_pieces; // [nme.name_piece(m,i)] -> bool

    friend class SearcherFactory;
public:
    virtual ~Searcher() { }

    // returns the number of Searcher::log invocations
    uint64_t step(Shape<8> empty_area);

protected:
    // relating to the board
    // will be initialized by SearcherFactory::run
    // before calling Searcher::log
    uint64_t config_index;

    explicit Searcher(bool sc = false)
        : shortcut{ sc },
          n_used_pieces{},
          used_pieces(g_nme->size_pieces(), 0) { }

    virtual bool log(uint64_t v) = 0;
};

// g_nme and g_sym must be set
// required before calling Searcher::Searcher
void compute_fast_canonical_form();

class SearcherFactory {
    uint64_t configs_issue_counter{};
    std::atomic<uint64_t> work_counter, configs_counter;

public:
    virtual ~SearcherFactory() { }

    void run();
    void run1();

    [[nodiscard]] uint64_t work_done() const {
        return work_counter.load(std::memory_order_relaxed);
    }
    [[nodiscard]] uint64_t configs_done() const {
        return configs_counter.load(std::memory_order_relaxed);
    }
    [[nodiscard]] uint64_t configs_issued() const {
        return configs_issue_counter;
    }

protected:
    // caller (SearcherFactory) is responsible for calling delete
    virtual Searcher *make() = 0;

    // whether the i-th board config should run
    [[nodiscard]] virtual bool should_run(uint64_t i, Shape<8> sh) { return true; }

    void incr_work(uint64_t diff = 1) {
        work_counter.fetch_add(diff, std::memory_order_relaxed);
    }

    virtual void after_run(uint64_t i, Shape<8> sh, uint64_t cnt, uint64_t ms);
};
