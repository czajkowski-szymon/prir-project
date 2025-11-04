#include <iostream>
#include <algorithm>
#include <limits>
#include <vector>
#include <random>
#include <chrono>

using namespace std;

// Implementacje sekwencyjne

template<typename T>
T sequential_sum(const vector<T>& vector) {
    T sum = 0;
    for (auto& element : vector) {
        sum += element;
    }

    return sum;
}

template<typename T>
T sequential_min(const vector<T>& vector) {
    T minimum = numeric_limits<T>::max();

    for (auto& element : vector) {
        minimum = min(minimum, element);
    }

    return minimum;
}

template<typename T>
T sequential_max(const vector<T>& vector) {
    T maximum = numeric_limits<T>::min();

    for (auto& element : vector) {
        maximum = max(maximum, element);
    }

    return maximum;
}

// Implementacje rownolegle

template<typename T>
T parallel_sum(const vector<T>& vector) {
    T sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (auto& element : vector) {
        sum += element;
    }

    return sum;
}

template<typename T>
T parallel_min(const vector<T>& vector) {
    T minimum = numeric_limits<T>::max();

    #pragma omp parallel for reduction(min:minimum)
    for (auto& element : vector) {
        minimum = min(minimum, element);
    }

    return minimum;
}

template<typename T>
T parallel_max(const vector<T>& vector) {
    T maximum = numeric_limits<T>::min();
    
    #pragma omp parallel for reduction(max:maximum)
    for (auto& element : vector) {
        maximum = max(maximum, element);
    }

    return maximum;
}

int main() {
    const size_t N = 1 << 26;
    
    vector<int64_t> vector(N);

    random_device rd;
    mt19937_64 gen(rd());
    uniform_int_distribution<int64_t> dist(0, 1000000);

    for (auto &x : vector) x = dist(gen);


    cout << "=====================" << endl;
    cout << "WYKONANIE SEKWENCYJNE" << endl;
    cout << "=====================" << endl;

    
    // SUMA - SEKWENCYJNIE
    const auto start_seq_sum = chrono::high_resolution_clock::now();
    auto seq_sum = sequential_sum(vector);
    const auto end_seq_sum = chrono::high_resolution_clock::now();
    chrono::duration<double> seq_time_sum = end_seq_sum - start_seq_sum;

    cout << "Sum of all elements is equal: " << seq_sum << endl;
    cout << "Execution time for sequential method: " << seq_time_sum.count() * 1000 << "ms" << endl;

    
    // MIN - SEKWENCYJNIE
    const auto start_seq_min = chrono::high_resolution_clock::now();
    auto seq_min = sequential_min(vector);
    const auto end_seq_min = chrono::high_resolution_clock::now();
    chrono::duration<double> seq_time_min = end_seq_min - start_seq_min;

    cout << "Minumum element of all elements is equal: " << seq_min << endl;
    cout << "Execution time for sequential method: " << seq_time_min.count() * 1000 << "ms" << endl;

    
    // MIN - SEKWENCYJNIE
    const auto start_seq_max = chrono::high_resolution_clock::now();
    auto seq_max = sequential_max(vector);
    const auto end_seq_max = chrono::high_resolution_clock::now();
    chrono::duration<double> seq_time_max = end_seq_max - start_seq_max;

    cout << "Maximum element of all elements is equal: " << seq_max << endl;
    cout << "Execution time for sequential method: " << seq_time_max.count() * 1000 << "ms" << endl;


    cout << "====================" << endl;
    cout << "WYKONANIE ROWNOLEGLE" << endl;
    cout << "====================" << endl;

    // SUMA - ROWNOLEGLE
    const auto start_par_sum = chrono::high_resolution_clock::now();
    auto par_sum = parallel_sum(vector);
    const auto end_par_sum = chrono::high_resolution_clock::now();
    chrono::duration<double> par_time_sum = end_par_sum - start_par_sum;

    cout << "Sum of all elements is equal: " << par_sum << endl;
    cout << "Execution time for parallel method: " << par_time_sum.count() * 1000 << "ms" << endl;

    
    // MIN - ROWNOLEGLE
    const auto start_par_min = chrono::high_resolution_clock::now();
    auto par_min = parallel_min(vector);
    const auto end_par_min = chrono::high_resolution_clock::now();
    chrono::duration<double> par_time_min = end_par_min - start_par_min;

    cout << "Sum of all elements is equal: " << par_min << endl;
    cout << "Execution time for parallel method: " << par_time_min.count() * 1000 << "ms" << endl;

    
    // SUMA - ROWNOLEGLE
    const auto start_par_max = chrono::high_resolution_clock::now();
    auto par_max = parallel_max(vector);
    const auto end_par_max = chrono::high_resolution_clock::now();
    chrono::duration<double> par_time_max = end_par_max - start_par_max;

    cout << "Sum of all elements is equal: " << par_max << endl;
    cout << "Execution time for parallel method: " << par_time_max.count() * 1000 << "ms" << endl;

    return 0;
}