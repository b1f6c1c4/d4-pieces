#pragma once

#include <cuda.h>
#include <algorithm>
#include <deque>
#include <format>
#include <functional>
#include <iostream>
#include <numeric>
#include <ranges>
#include <vector>

#include "util.hpp"
#include "growable.h"

#define C(ans) { chk_impl((ans), __FILE__, __LINE__); }
static void chk_impl(CUresult code, const char *file, int line);

template <typename T>
class Growable {
    using value_type = T;
    using R = Rg<T>;

    struct RH : R {
        CUmemGenericAllocationHandle h;
    };

    // in units of T
    std::deque<R> vmaps; // sum(vmaps, &R::len) == reserved
    size_t reserved; // offset + mapped <= reserved
    size_t offset;
    size_t used; // used <= mapped
    // vmaps[0].ptr + offset == maps[0].ptr
    std::deque<RH> maps; // sum(maps, &R::len) == mapped
    size_t mapped;
    std::vector<R> evicted_data;
    size_t evicted;
    size_t chunk; // granularity

    CUmemAllocationProp prop;
    CUmemAccessDesc adesc;

public:
    explicit Growable(int dev, size_t max = 0, size_t c = 1);
    Growable &operator=(Growable &&other) noexcept;
    ~Growable();

    static_assert(std::is_trivially_constructible_v<T>, "T not trivially constructible");
    static_assert(std::is_trivially_copyable_v<T>, "T not trivially copyable");

    // return how many T can be written without crash
    [[nodiscard]] size_t risk_free_size() const { return mapped - used; }
    [[nodiscard]] size_t get_load() const { return used + evicted; }
    // re-organize all pa mappings s.t. reserved >= offset + new_reserved
    void remap(size_t new_max, bool force = false);
    // mark n of Ts are actually consumed
    void commit(size_t n) { used += n; }
    // free up unused pa
    void compact();
    // allocate a contiguous T[n]
    T *get(size_t n) {
        if (ensure(n))
            return vmaps[0].ptr + used;
        return nullptr;
    }

    void mem_stat() const;

    std::vector<R> &&remove_data() {
        evicted = 0;
        return std::move(evicted_data);
    }

    // make sure risk_free_size() >= n, and return the write-start point
    [[nodiscard]] bool ensure(size_t n);

    // copy all useful data from the 0-th pa to evicted_data
    // does NOT free up pa
    void evict1();

    // copy all useful data to evicted_data
    // does NOT free up pa
    void evict_all();

    // remove unused vmap
    void cleanup();
};

template <typename T>
Growable<T>::Growable(int dev, size_t max, size_t c)
    : reserved{}, vmaps{}, offset{}, used{}, maps{},
      mapped{}, evicted_data{}, evicted{}, chunk{},
      prop{}, adesc{} {
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = dev;
    adesc.location = prop.location;
    adesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    C(cuMemGetAllocationGranularity(&chunk, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    chunk = std::lcm(chunk, c * sizeof(T)) / sizeof(T);
    remap(max);
};

template <typename T>
Growable<T> &Growable<T>::operator=(Growable<T> &&other) noexcept {
    this->vmaps = std::move(other.vmaps);
    this->reserved = std::move(other.reserved);
    this->offset = std::move(other.offset);
    this->used = std::move(other.used);
    this->maps = std::move(other.maps);
    this->mapped = std::move(other.mapped);
    this->evicted_data = std::move(other.evicted_data);
    this->evicted = std::move(other.evicted);
    this->chunk = other.chunk;
    this->prop = other.prop;
    this->adesc = other.adesc;
    other.reserved = 0;
    other.offset = 0;
    other.used = 0;
    other.evicted = 0;
}

template <typename T>
Growable<T>::~Growable() {
    for (auto rh : maps) {
        C(cuMemUnmap((CUdeviceptr)rh.ptr, rh.len * sizeof(T)));
        C(cuMemRelease(rh.h));
    }
    for (auto v : vmaps)
        C(cuMemAddressFree((CUdeviceptr)v.ptr, v.len * sizeof(T)));
    for (auto r : evicted_data)
        delete [] r.ptr;
    evicted_data.clear();
}

template <typename T>
bool Growable<T>::ensure(size_t n) {
    if (mapped && used + n <= mapped) return true;

    auto sz = std::max((used + n - mapped + chunk - 1) / chunk, 1zu) * chunk;
    remap(mapped + sz); // vmap must not be empty

    CUmemGenericAllocationHandle h;
    CUresult err;
    if ((err = cuMemCreate(&h, sz * sizeof(T), &prop, 0)) == CUDA_SUCCESS) goto map;
    if (err != CUDA_ERROR_OUT_OF_MEMORY) C(err);
    if (!mapped) goto fail;
again:
    evict1();
    if (used + n <= mapped) return true;
    sz = std::max((used + n - mapped + chunk - 1) / chunk, 1zu) * chunk;
    if ((err = cuMemCreate(&h, sz * sizeof(T), &prop, 0)) == CUDA_SUCCESS) goto map;
    if (used && err == CUDA_ERROR_OUT_OF_MEMORY) goto again;
    if (err != CUDA_ERROR_OUT_OF_MEMORY) C(err);
fail:
    evict_all();
    return false;
map:
    auto ptr = vmaps[0].ptr + offset + mapped;
    C(cuMemMap((CUdeviceptr)ptr, sz * sizeof(T), 0, h, 0));
    C(cuMemSetAccess((CUdeviceptr)ptr, sz * sizeof(T), &adesc, 1));
    maps.emplace_back(RH{ R{ ptr, sz }, h });
    mapped += sz;
    return true;
}

template <typename T>
void Growable<T>::evict1() {
    if (maps.empty())
        throw std::runtime_error{ "cannot evict: nothing was allocated" };

    auto src = maps.front();
    auto used1 = min((unsigned long long)used, src.len);
    if (used1) {
        auto dst = evicted_data.emplace_back(R{ new T[used1], used1 });
        if (!dst.ptr)
            throw std::runtime_error{ std::format("new T[{}] failed ({} MiB)", used1, used1 * sizeof(T) / 1048576.0) };
        std::cout << std::format("dev#{}: cuMemcpyDtoH {}B from 0x{:016x} => 0x{:016x}\n",
                prop.location.id, display(used1 * sizeof(T)), (size_t)src.ptr, (size_t)dst.ptr);
        auto err = cuMemcpyDtoH(dst.ptr, (CUdeviceptr)src.ptr, used1 * sizeof(T));
        if (err != CUDA_SUCCESS) {
            mem_stat();
            C(err);
        }
        evicted += used1;
    }
    if (used < src.len) {
        used = 0;
    } else {
        // remap(mapped + src.len);
        maps.pop_front();
        C(cuMemUnmap((CUdeviceptr)src.ptr, src.len * sizeof(T)));
        src.ptr += mapped;
        offset += src.len;
        if (src.ptr + src.len >= vmaps[0].ptr + reserved) {
            std::cout << std::format("WARNING: rotation failed as vmem exhausted");
            C(cuMemRelease(src.h));
            mapped -= src.len;
        } else {
            C(cuMemMap((CUdeviceptr)src.ptr, src.len * sizeof(T), 0, src.h, 0));
            C(cuMemSetAccess((CUdeviceptr)src.ptr, src.len * sizeof(T), &adesc, 1));
            maps.push_back(src);
        }
        used -= src.len;
    }
}

template <typename T>
void Growable<T>::compact() {
    if (maps.empty()) {
        cleanup();
        return;
    }
    auto beg = std::ranges::lower_bound(maps, maps[0].ptr + used, std::less{}, &RH::ptr);
    for (auto it = beg; it != maps.end(); it++) {
        C(cuMemUnmap((CUdeviceptr)it->ptr, it->len * sizeof(T)));
        mapped -= it->len;
        C(cuMemRelease(it->h));
    }
    maps.erase(beg, maps.end());
    cleanup();
}

template <typename T>
void Growable<T>::evict_all() {
    if (maps.empty() || !used)
        return;
    auto dst = evicted_data.emplace_back(R{ new T[used], used });
    if (!dst.ptr)
        throw std::runtime_error{ std::format("new T[{}] failed ({} MiB)", used, used * sizeof(T) / 1048576.0) };
    C(cuMemcpyDtoH(dst.ptr, (CUdeviceptr)maps[0].ptr, used * sizeof(T)));
    evicted += used;
    used = 0;
}

template <typename T>
void Growable<T>::remap(size_t new_max, bool force) {
    new_max = (new_max + chunk - 1) / chunk * chunk;
    if (!force && offset + new_max <= reserved)
        return;

    cleanup();

    CUdeviceptr new_ptr{};
    if (!force && !vmaps.empty()
            && cuMemAddressReserve(&new_ptr, (new_max - reserved) * sizeof(T), 
                alignof(T), (CUdeviceptr)(vmaps[0].ptr + reserved), 0) == CUDA_SUCCESS
            && new_ptr == (CUdeviceptr)(vmaps[0].ptr + reserved)) {
        vmaps.emplace_back(R{ (T *)new_ptr, new_max - reserved }); 
        reserved = new_max;
        return;
    }
    if (new_ptr) { // remove accidentally created vmap
        C(cuMemAddressFree(new_ptr, (new_max - reserved) * sizeof(T)));
    }
    C(cuMemAddressReserve(&new_ptr, new_max * sizeof(T), alignof(T), 0, 0));
    offset = 0;
    auto o = (T *)new_ptr;
    for (auto &rh : maps) {
        C(cuMemUnmap((CUdeviceptr)rh.ptr, rh.len * sizeof(T)));
        rh.ptr = o;
        C(cuMemMap((CUdeviceptr)rh.ptr, rh.len * sizeof(T), 0, rh.h, 0));
        C(cuMemSetAccess((CUdeviceptr)rh.ptr, rh.len * sizeof(T), &adesc, 1));
        o += rh.len;
    }
    for (auto vm : vmaps)
        C(cuMemAddressFree((CUdeviceptr)vm.ptr, vm.len * sizeof(T)));
    vmaps.clear();
    vmaps.emplace_back(R{ (T *)new_ptr, new_max });
    reserved = new_max;
}

template <typename T>
void Growable<T>::cleanup() {
    while (!vmaps.empty() && vmaps.front().len <= offset) {
        auto vm = vmaps.front();
        C(cuMemAddressFree((CUdeviceptr)vm.ptr, vm.len * sizeof(T)));
        offset -= vm.len;
        reserved -= vm.len;
        vmaps.pop_front();
    }
    while (!vmaps.empty() && vmaps.back().len <= reserved - offset - mapped) {
        auto vm = vmaps.back();
        C(cuMemAddressFree((CUdeviceptr)vm.ptr, vm.len * sizeof(T)));
        reserved -= vm.len;
        vmaps.pop_back();
    }
}

template <typename T>
void Growable<T>::mem_stat() const {
    std::cout << std::format(R"(
chunk:     {:10} = {}
reserved:  {:10} = {} ({} vmaps)
offset:    {:10} = {}
used:      {:10} = {}
mapped:    {:10} = {} ({} maps)
evicted:   {:10} = {} ({} pieces)
risk-free: {:10} = {}
)",
            display(chunk * sizeof(T)), chunk,
            display(reserved * sizeof(T)), reserved, vmaps.size(),
            display(offset * sizeof(T)), offset,
            display(used * sizeof(T)), used,
            display(mapped * sizeof(T)), mapped, maps.size(),
            display(evicted * sizeof(T)), evicted, evicted_data.size(),
            display(risk_free_size() * sizeof(T)), risk_free_size());
    for (auto vm : vmaps)
        std::cout << std::format("  vmaps[0x{:016x}:{:016x}) => {}\n",
                (ptrdiff_t)vm.ptr, (ptrdiff_t)(vm.ptr + vm.len), display(vm.len * sizeof(T)));
    for (auto rh : maps)
        std::cout << std::format("    maps[0x{:016x}:{:016x}) => {}\n",
                (ptrdiff_t)rh.ptr, (ptrdiff_t)(rh.ptr + rh.len), display(rh.len * sizeof(T)));
}

/*
int main() {
    Growable<float> gr{};
    std::string str;
    double sz;
    float *ptr{};
    while (true) {
        gr.mem_stat();
        if (ptr)
            std::cout << std::format("  ptr => 0x{:016x}\n", (uint64_t)ptr);
        std::cin >> str;
        if (str == "r") {
            std::cin >> sz;
            gr.remap((size_t)sz);
        } else if (str == "rf") {
            std::cin >> sz;
            gr.remap((size_t)sz, true);
        } else if (str == "cm" || str == "c") {
            std::cin >> sz;
            curandGenerator_t gen;
            C(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
            C(curandSetPseudoRandomGeneratorSeed(gen, (size_t)sz));
            C(curandGenerateUniform(gen, ptr, (size_t)sz));
            ptr += (size_t)sz;
            gr.commit((size_t)sz);
        } else if (str == "en" || str == "e") {
            std::cin >> sz;
            ptr = gr.get((size_t)sz);
        } else if (str == "x") {
            gr.compact();
        } else if (str == "e1") {
            gr.evict1();
        } else if (str == "ea") {
            gr.evict_all();
        } else if (str == "cl") {
            gr.cleanup();
        }
    }
}
*/
