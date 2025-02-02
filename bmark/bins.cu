#include <cuda.h>
#include <curand.h>
#include <algorithm>
#include <iostream>
#include <format>
#include <deque>
#include <ranges>
#include <vector>

#define C(ans) { chk_impl((ans), __FILE__, __LINE__); }

static inline void chk_impl(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        throw std::runtime_error{
            std::format("CUDA: {}: {} @ {}:{}\n",
                    cudaGetErrorName(code), cudaGetErrorString(code),
                    file, line) };
    }
}

static inline void chk_impl(CUresult code, const char *file, int line) {
    const char *pn = "???", *ps = "???";
    cuGetErrorName(code, &pn);
    cuGetErrorString(code, &ps);
    if (code != CUDA_SUCCESS) {
        throw std::runtime_error{
            std::format("CUDA Driver: {}: {} @ {}:{}\n", pn, ps, file, line) };
    }
}

static inline void chk_impl(curandStatus_t code, const char *file, int line) {
    if (code != CURAND_STATUS_SUCCESS) {
        throw std::runtime_error{
            std::format("curand: {} @ {}:{}\n", (int)code, file, line) };
    }
}

template <typename T>
class Growable {
    struct R {
        T *ptr;
        size_t len; // number of T
    };
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

    CUmemAllocationProp prop, uprop;
    CUmemAccessDesc adesc, uadesc;

public:
    explicit Growable(size_t max = 0);
    ~Growable();

    static_assert(std::is_trivially_constructible_v<T>, "T not trivially constructible");
    static_assert(std::is_trivially_copyable_v<T>, "T not trivially copyable");

    // return how many T can be written without crash
    [[nodiscard]] size_t risk_free_size() const { return mapped - used; }
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

    // make sure risk_free_size() >= n, and return the write-start point
    bool ensure(size_t n);

    // copy all useful data from the 0-th pa to evicted_data
    // does NOT free up pa
    void evict1();

    // copy all useful data to evicted_data
    // free up all pa
    void evict_all();

    // remove unused vmap
    void cleanup();
};

template <typename T>
Growable<T>::Growable(size_t max)
    : reserved{}, vmaps{}, offset{}, used{}, maps{},
      mapped{}, evicted_data{}, evicted{}, chunk{},
      prop{}, uprop{}, adesc{}, uadesc{} {

    int n; C(cudaGetDeviceCount(&n)); // dark magic; don't touch

    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0; // TODO
    adesc.location = prop.location;
    adesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    uprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    uprop.location.type = CU_MEM_LOCATION_TYPE_HOST_NUMA;
    uprop.location.id = 0; // TODO
    uadesc.location = uprop.location;
    uadesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    C(cuMemGetAllocationGranularity(&chunk, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    chunk = (chunk + sizeof(T) - 1) / sizeof(T);
    remap(max);
};

template <typename T>
Growable<T>::~Growable() {
    for (auto rh : maps) {
        C(cuMemUnmap((CUdeviceptr)rh.ptr, rh.len * sizeof(T)));
        C(cuMemRelease(rh.h));
    }
    for (auto v : vmaps)
        C(cuMemAddressFree((CUdeviceptr)v.ptr, v.len * sizeof(T)));
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
    auto used1 = min(used, src.len);
    if (used1) {
        auto dst = evicted_data.emplace_back(R{ new T[used1], used1 });
        if (!dst.ptr)
            throw std::runtime_error{ std::format("new T[{}] failed ({} MiB)", used1, used1 * sizeof(T) / 1048576.0) };
        C(cudaMemcpy(dst.ptr, src.ptr, used1 * sizeof(T), cudaMemcpyDeviceToHost));
        evicted += used1;
    }
    if (used < src.len) {
        used = 0;
    } else {
        remap(mapped + src.len);
        src = maps.front();
        C(cuMemUnmap((CUdeviceptr)src.ptr, src.len * sizeof(T)));
        offset += src.len;
        C(cuMemMap((CUdeviceptr)src.ptr, src.len * sizeof(T), 0, src.h, 0));
        C(cuMemSetAccess((CUdeviceptr)src.ptr, src.len * sizeof(T), &adesc, 1));
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
    if (maps.empty())
        return;
    if (used) {
        auto dst = evicted_data.emplace_back(R{ new T[used], used });
        if (!dst.ptr)
            throw std::runtime_error{ std::format("new T[{}] failed ({} MiB)", used, used * sizeof(T) / 1048576.0) };
        C(cudaMemcpy(dst.ptr, maps[0].ptr, used * sizeof(T), cudaMemcpyDeviceToHost));
        evicted += used;
        used = 0;
    }
    for (auto rh : maps) {
        C(cuMemUnmap((CUdeviceptr)rh.ptr, rh.len * sizeof(T)));
        C(cuMemRelease(rh.h));
    }
    maps.clear();
    mapped = 0;
    for (auto vm : vmaps) {
        C(cuMemAddressFree((CUdeviceptr)vm.ptr, vm.len * sizeof(T)));
    }
    vmaps.clear();
    reserved = 0;
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

std::string display(uint64_t byte) {
    if (byte < 1000ull)
        return std::format("{}", byte);
    if (byte < 1024 * 1024ull)
        return std::format("{:.2f} Ki", 1.0 * byte / 1024);
    if (byte < 1024 * 1024ull * 1024ull)
        return std::format("{:.2f} Mi", 1.0 * byte / 1024 / 1024);
    if (byte < 1024 * 1024ull * 1024ull * 1024ull)
        return std::format("{:.2f} Gi", 1.0 * byte / 1024 / 1024 / 1024);
    return std::format("{:.3f} TiB", 1.0 * byte / 1024 / 1024 / 1024 / 1024);
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
