#include "searcher.hpp"

#include <chrono>
#include <print>

#define BOOST_THREAD_VERSION 5
#include <boost/thread/executors/basic_thread_pool.hpp>
#include <boost/thread/future.hpp>
#include <boost/container/small_vector.hpp>

#include "naming.inl"

std::optional<Board<8>> g_board;
std::optional<Naming> g_nme;
unsigned g_sym;

// Shape -> piece naming
// use 240MiB of memory to guarantee no hash collision
static constexpr size_t fast_canonical_form_mod = 30878073; // dark magic; don't touch
static uint64_t fast_canonical_form[fast_canonical_form_mod]; // bss, zero-initialized
void compute_fast_canonical_form() {
    auto count = 0zu;
    auto ttcount = 0zu;
    auto push_translate = [&](uint64_t nm, Shape<8> t) {
        for (auto row = 0u; row <= t.bottom(); row++)
            for (auto col = 0u; col <= t.right(); col++) {
                ttcount++;
                auto v = t.translate_unsafe(col, row).get_value();
                auto &elem = fast_canonical_form[v % fast_canonical_form_mod];
                // note: hash collision on nm == 0 are undiscoverable
                if (elem && elem != nm)
                    throw std::runtime_error{ "panic: dark magic doesn't work!" };
                elem = nm;
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
    std::print("cached {}/{} => {} canonical forms\n", ttcount, fast_canonical_form_mod, count);
    // the code searching for dark magic
    /*
    std::atomic<uint64_t> best{ ~0ull };
    {
        std::vector<std::jthread> thrs;
        for (auto i = 0; i < 128; i++)
            thrs.emplace_back([&](int seed) {
                std::mt19937_64 rnd{};
                rnd.seed(seed);
                std::uniform_int_distribution<uint64_t> dist(map.size(), 30878073ull);
                std::vector<char> fm(30878073ull, 0);
            bad:
                auto mod = dist(rnd);
                std::memset(fm.data(), 0, mod);
                for (auto [k, v] : map) {
                    if (fm[k.get_value() % mod])
                        goto bad;
                    fm[k.get_value() % mod] = true;
                }
                auto old = best.load();
                if (mod >= old) goto bad;
                while (best.compare_exchange_weak(old, mod, std::memory_order_relaxed))
                    if (mod >= old) goto bad;
                std::print("found: {}/{} => {} canonical forms\n", map.size(), mod, count);
                goto bad;
            }, i);
    }
    std::print("cached {}/{} => {} canonical forms\n", map.size(), fast_canonical_form_mod, count);
    std::abort();
    */
}

using sv_t = boost::container::small_vector<uint64_t, 24zu>;

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
    auto recurse = [&,this](Shape<8> sh) {
        auto &pcs = used_pieces[fast_canonical_form[sh.get_value() % fast_canonical_form_mod]];
        if (pcs)
            return false; // duplicate shape
        pcs = 1, n_used_pieces++;
        cnt += step(empty_area - sh);
        pcs = 0, n_used_pieces--;
        return cnt && shortcut;
    };
    // all possible shapes of size m covering front()
    sv_t shapes{ empty_area.front_shape().get_value() };
    if (1u >= g_nme->min_m)
        if (recurse(empty_area.front_shape()))
            return cnt;
    for (auto m = 1u; m < g_nme->max_m; m++) {
        sv_t next;
        next.reserve(4 * shapes.size());
        for (auto shv : shapes) {
            Shape<8> sh{ shv };
            // the following is equivalent to this, but is faster:
            //   for (auto pos : (sh.extend1() & empty_area) - sh)
            //       next.push_back(sh.set(pos).get_value());
            auto ex = (sh.extend1() & empty_area) - sh;
            for (auto v = ex.get_value(); v; v -= (v & -v)) [[likely]]
                next.push_back(shv | (v & -v));
        }
        if (next.empty())
            break;
        std::ranges::sort(next);
        next.erase(std::ranges::unique(next).begin(), next.end());
        if (m + 1 >= g_nme->min_m)
            for (auto sh : next)
                if (recurse(Shape<8>{ sh }))
                    return cnt;
        shapes = std::move(next);
    }
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
