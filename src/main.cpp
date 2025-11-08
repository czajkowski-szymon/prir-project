#include <iostream>
#include <algorithm>
#include <limits>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>

#define THREADS 12

using namespace std;

// Implementacje sekwencyjne

template<typename T>
T sequential_sum(const vector<T>& in) {
    T sum = 0;

    for (auto& element : in) {
        sum += element;
    }

    return sum;
}

template<typename T>
T sequential_min(const vector<T>& in) {
    T minimum = numeric_limits<T>::max();

    for (auto& element : in) {
        minimum = min(minimum, element);
    }

    return minimum;
}

template<typename T>
T sequential_max(const vector<T>& in) {
    T maximum = numeric_limits<T>::min();

    for (auto& element : in) {
        maximum = max(maximum, element);
    }

    return maximum;
}

template<typename T>
vector<T> sequential_exclusive_scan(const vector<T>& in) {
    size_t n = in.size();
    if (n == 0) return {};

    vector<T> result(n);
    result[0] = 0;

    for (size_t i = 1; i < n; i++)
        result[i] = result[i - 1] + in[i - 1];

    return result;
}


// Implementacje rownolegle

template<typename T>
T parallel_sum(const vector<T>& in) {
    T sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (auto& element : in) {
        sum += element;
    }

    return sum;
}

template<typename T>
T parallel_min(const vector<T>& in) {
    T minimum = numeric_limits<T>::max();

    #pragma omp parallel for reduction(min:minimum)
    for (auto& element : in) {
        minimum = min(minimum, element);
    }

    return minimum;
}

template<typename T>
T parallel_max(const vector<T>& in) {
    T maximum = numeric_limits<T>::min();

    #pragma omp parallel for reduction(max:maximum)
    for (auto& element : in) {
        maximum = max(maximum, element);
    }

    return maximum;
}

template<typename T>
void blelloch_scan(vector<T>& in) {
    size_t N = in.size();
    if (N == 0) return;

    size_t levels = log2(N);

    for (size_t i = 0; i < levels; i++) {
        size_t step = 1 << (i + 1);
        size_t offset = 1 << i;

        #pragma omp parallel for schedule(static)
        for (size_t j = step - 1; j < N; j += step) {
            in[j] += in[j - offset];
        }
    }

    in[N - 1] = 0;

    for (int i = levels - 1; i >= 0; i--) {
        size_t step = 1 << (i + 1);
        size_t offset = 1 << i;

        #pragma omp parallel for schedule(static)
        for (size_t j = step - 1; j < N; j += step) {
            T temp = in[j - offset];
            in[j - offset] = in[j];
            in[j] += temp;
        }
    }
}

int main() {
    const size_t N = 1 << 24;
    
    vector<int64_t> input_vector(N);

    random_device rd;
    mt19937_64 gen(rd());
    uniform_int_distribution<int64_t> dist(0, 1000000);

    for (auto &x : input_vector) x = dist(gen);


    cout << "=====================" << endl;
    cout << "WYKONANIE SEKWENCYJNE" << endl;
    cout << "=====================" << endl;

    
    // SUMA - SEKWENCYJNIE
    const auto start_seq_sum = chrono::high_resolution_clock::now();
    auto seq_sum = sequential_sum(input_vector);
    const auto end_seq_sum = chrono::high_resolution_clock::now();
    chrono::duration<double> seq_time_sum = end_seq_sum - start_seq_sum;

    cout << "Sum of all elements is equal: " << seq_sum << endl;
    cout << "Execution time for sum reduction sequential method: " << seq_time_sum.count() * 1000 << "ms" << endl << endl;

    
    // MIN - SEKWENCYJNIE
    const auto start_seq_min = chrono::high_resolution_clock::now();
    auto seq_min = sequential_min(input_vector);
    const auto end_seq_min = chrono::high_resolution_clock::now();
    chrono::duration<double> seq_time_min = end_seq_min - start_seq_min;

    cout << "Minumum element of all elements is equal: " << seq_min << endl;
    cout << "Execution time for min reduction sequential method: " << seq_time_min.count() * 1000 << "ms" << endl << endl;

    
    // MAX - SEKWENCYJNIE
    const auto start_seq_max = chrono::high_resolution_clock::now();
    auto seq_max = sequential_max(input_vector);
    const auto end_seq_max = chrono::high_resolution_clock::now();
    chrono::duration<double> seq_time_max = end_seq_max - start_seq_max;

    cout << "Maximum element of all elements is equal: " << seq_max << endl;
    cout << "Execution time for max reduction sequential method: " << seq_time_max.count() * 1000 << "ms" << endl << endl;

    // PREFIX-SUM - SEKWENCYJNIE
    const auto start_seq_prefixsum = chrono::high_resolution_clock::now();
    auto seq_prefixsum = sequential_exclusive_scan(input_vector);
    const auto end_seq_prefixsum = chrono::high_resolution_clock::now();
    chrono::duration<double> seq_time_prefixsum = end_seq_prefixsum - start_seq_prefixsum;

    cout << "Execution time for prefix-sum scan sequential method: " << seq_time_prefixsum.count() * 1000 << "ms" << endl << endl;


    cout << "====================" << endl;
    cout << "WYKONANIE ROWNOLEGLE" << endl;
    cout << "====================" << endl;

    // SUMA - ROWNOLEGLE
    const auto start_par_sum = chrono::high_resolution_clock::now();
    auto par_sum = parallel_sum(input_vector);
    const auto end_par_sum = chrono::high_resolution_clock::now();
    chrono::duration<double> par_time_sum = end_par_sum - start_par_sum;

    cout << "Sum of all elements is equal: " << par_sum << endl;
    cout << "Execution time for sum reduction parallel method: " << par_time_sum.count() * 1000 << "ms" << endl << endl;

    
    // MIN - ROWNOLEGLE
    const auto start_par_min = chrono::high_resolution_clock::now();
    auto par_min = parallel_min(input_vector);
    const auto end_par_min = chrono::high_resolution_clock::now();
    chrono::duration<double> par_time_min = end_par_min - start_par_min;

    cout << "Sum of all elements is equal: " << par_min << endl;
    cout << "Execution time for min reduction parallel method: " << par_time_min.count() * 1000 << "ms" << endl << endl;

    
    // MAX - ROWNOLEGLE
    const auto start_par_max = chrono::high_resolution_clock::now();
    auto par_max = parallel_max(input_vector);
    const auto end_par_max = chrono::high_resolution_clock::now();
    chrono::duration<double> par_time_max = end_par_max - start_par_max;

    cout << "Sum of all elements is equal: " << par_max << endl;
    cout << "Execution time for max reduction parallel method: " << par_time_max.count() * 1000 << "ms" << endl << endl;

    // PREFIX-SUM - ROWNOLEGLE
    const auto start_par_prefixsum = chrono::high_resolution_clock::now();
    blelloch_scan(input_vector);
    const auto end_par_prefixsum = chrono::high_resolution_clock::now();
    chrono::duration<double> par_time_prefixsum = end_par_prefixsum - start_par_prefixsum;

    cout << "Execution time for prefix-sum scan parallel method: " << par_time_prefixsum.count() * 1000 << "ms" << endl;

    return 0;
}