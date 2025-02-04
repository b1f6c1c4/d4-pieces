#pragma once

#include <string>

inline std::string display(uint64_t byte) {
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
