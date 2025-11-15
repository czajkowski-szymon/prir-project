#pragma once
#include <vector>
#include <limits>
#include <algorithm>

template<typename T>
T sequential_sum(const std::vector<T>& in) {
    T sum = 0;
    for (const auto& element : in) sum += element;
    return sum;
}

template<typename T>
T sequential_min(const std::vector<T>& in) {
    T minimum = std::numeric_limits<T>::max();
    for (const auto& element : in) minimum = std::min(minimum, element);
    return minimum;
}

template<typename T>
T sequential_max(const std::vector<T>& in) {
    T maximum = std::numeric_limits<T>::min();
    for (const auto& element : in) maximum = std::max(maximum, element);
    return maximum;
}

template<typename T>
std::vector<T> sequential_exclusive_scan(const std::vector<T>& in) {
    size_t n = in.size();
    if (n == 0) return {};
    std::vector<T> result(n);
    result[0] = 0;
    for (size_t i = 1; i < n; ++i) result[i] = result[i - 1] + in[i - 1];
    return result;
}