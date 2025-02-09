#pragma once

#if defined(__cpp_lib_print)
#include <print>
#else
#include <iostream>
#endif
#include <format>
#include <concepts>

template <std::integral T>
inline std::string display(T byte) {
    if (byte < 1000ull)
        return std::format("{}", byte);
    if (byte < 1024 * 1024ull)
        return std::format("{:.2f} Ki", 1.0 * byte / 1024);
    if (byte < 1024 * 1024ull * 1024ull)
        return std::format("{:.2f} Mi", 1.0 * byte / 1024 / 1024);
    if (byte < 1024 * 1024ull * 1024ull * 1024ull)
        return std::format("{:.2f} Gi", 1.0 * byte / 1024 / 1024 / 1024);
    return std::format("{:.3f} Ti", 1.0 * byte / 1024 / 1024 / 1024 / 1024);
}

inline std::string display(double s) {
    if (s == 0.0)
        return "0s";
    if (s < 0.0)
        return "-" + display(-s);
    if (s < 1e-9)
        return std::format("<1ns");
    if (s < 1e-6)
        return std::format("{:.1f}ns", s / 1e-9);
    if (s < 1e-3)
        return std::format("{:.1f}us", s / 1e-6);
    if (s < 1.0)
        return std::format("{:.1f}ms", s / 1e-3);
    if (s < 100.0)
        return std::format("{:.2f}s", s);
    if (s < 600.0)
        return std::format("{}m{}s", (uint64_t)(s) / 60, (uint64_t)(s) % 60);
    if (s < 3600.0)
        return std::format("{:.1f}m", s / 60);
    if (s < 100 * 3600.0)
        return std::format("{}h{}m", (uint64_t)(s) / 3600, (uint64_t)(s) / 60 % 60);
    if (s < 100 * 86400.0)
        return std::format("{}d{}h", (uint64_t)(s) / 86400, (uint64_t)(s) / 3600 % 24);
    return std::format("{}d", (uint64_t)(s) / 86400);
}

constexpr inline auto count_digits(unsigned long long v) {
    if (v <= 9) return 1ull;
    return 1ull + count_digits(v / 10);
}

// poly-fill for stupid g++-13
#if !defined(__cpp_lib_print)
namespace std {
    template <typename ... TArgs>
    void print(std::format_string<TArgs...> fmt, TArgs && ... args) {
        std::cout << std::format(fmt, std::forward<TArgs>(args)...);
    }
    template <typename ... TArgs>
    void print(std::ostream &os, std::format_string<TArgs...> fmt, TArgs && ... args) {
        os << std::format(fmt, std::forward<TArgs>(args)...);
    }
}
#endif
