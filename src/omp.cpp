#include <iostream>
#include <algorithm>
#include <limits>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include "sequential.hpp"
#include <fstream>
#include <numeric>
#include <cmath>

#define THREADS 12

using namespace std;

// Implementacje rownolegle
template<typename T>
T parallel_sum(const vector<T>& in) {
    T sum = 0;

    // #pragma omp parallel for reduction(+:sum)
    #pragma omp parallel for reduction(+:sum) schedule(runtime)
    for (auto& element : in) {
        sum += element;
    }

    return sum;
}

template<typename T>
T parallel_min(const vector<T>& in) {
    T minimum = numeric_limits<T>::max();

    // #pragma omp parallel for reduction(min:minimum)
    #pragma omp parallel for reduction(min:minimum) schedule(runtime)
    for (auto& element : in) {
        minimum = min(minimum, element);
    }

    return minimum;
}

template<typename T>
T parallel_max(const vector<T>& in) {
    T maximum = numeric_limits<T>::min();

    // #pragma omp parallel for reduction(max:maximum)
    #pragma omp parallel for reduction(max:maximum) schedule(runtime)
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

        // #pragma omp parallel for schedule(static)
        #pragma omp parallel for schedule(runtime)
        for (size_t j = step - 1; j < N; j += step) {
            in[j] += in[j - offset];
        }
    }

    in[N - 1] = 0;

    for (int i = levels - 1; i >= 0; i--) {
        size_t step = 1 << (i + 1);
        size_t offset = 1 << i;

        // #pragma omp parallel for schedule(static)
        #pragma omp parallel for schedule(runtime)
        for (size_t j = step - 1; j < N; j += step) {
            T temp = in[j - offset];
            in[j - offset] = in[j];
            in[j] += temp;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <type: int|double> <operation: sum|min|max|prefix> [power] [threads] [chunk]" << endl;
        return 1;
    }
    string type_str = argv[1] ? argv[1] : "double";
    string op_str = argv[2] ? argv[2] : "sum";
    string power = argv[3] ? argv[3] : "20";
    string threads_arg = argv[4] ? argv[4] : to_string(THREADS);
    string chunk_arg = argv[5] ? argv[5] : "1";

    const size_t N = 1 << stoi(power);

    int threads = stoi(threads_arg);
    int chunk = stoi(chunk_arg);
    omp_set_num_threads(threads);
    omp_set_schedule(omp_sched_static, chunk);

    if (type_str == "int") {
        vector<int64_t> data(N);
        mt19937_64 gen(1234);
        uniform_int_distribution<int64_t> dist(0, 1000000);
        for (auto &x : data) x = dist(gen);

        if (op_str == "sum") {
            const auto t0 = chrono::high_resolution_clock::now();
            auto seq = sequential_sum(data);
            const auto t1 = chrono::high_resolution_clock::now();
            chrono::duration<double> seq_time = t1 - t0;

            const auto p0 = chrono::high_resolution_clock::now();
            auto par = parallel_sum(data);
            const auto p1 = chrono::high_resolution_clock::now();
            chrono::duration<double> par_time = p1 - p0;

            double abs_err = fabs((double)seq - (double)par);
            double rel_err = fabs(seq) > 0 ? abs_err / fabs((double)seq) : 0.0;

            cout << "[SEQ] Sum: " << seq << " time(ms): " << seq_time.count()*1000 << endl;
            cout << "[PAR] Sum: " << par << " time(ms): " << par_time.count()*1000 << endl;
            cout << "[CONFIG] threads=" << threads << " chunk=" << chunk << endl;

            // string filename = "../results/openmp/openmp_sum_int.csv";
            string filename = "../results/openmp/chunks/openmp_sum_int.csv";

            ofstream fout(filename, ios::app);
            fout << power << "," << threads << "," << par_time.count()*1000 << "," << abs_err << "," << rel_err << "," << chunk << endl;
            fout.close();
        } else if (op_str == "min") {
            const auto t0 = chrono::high_resolution_clock::now();
            auto seq = sequential_min(data);
            const auto t1 = chrono::high_resolution_clock::now();
            chrono::duration<double> seq_time = t1 - t0;

            const auto p0 = chrono::high_resolution_clock::now();
            auto par = parallel_min(data);
            const auto p1 = chrono::high_resolution_clock::now();
            chrono::duration<double> par_time = p1 - p0;

            double abs_err = fabs((double)seq - (double)par);
            double rel_err = fabs(seq) > 0 ? abs_err / fabs((double)seq) : 0.0;

            cout << "[SEQ] Min: " << seq << " time(ms): " << seq_time.count()*1000 << endl;
            cout << "[PAR] Min: " << par << " time(ms): " << par_time.count()*1000 << endl;
            cout << "[CONFIG] threads=" << threads << " chunk=" << chunk << endl;

            // string filename = "../results/openmp/openmp_min_int.csv";
            string filename = "../results/openmp/chunks/openmp_min_int.csv";

            ofstream fout(filename, ios::app);
            fout << power << "," << threads << "," << par_time.count()*1000 << "," << abs_err << "," << rel_err << "," << chunk << endl;
            fout.close();
        } else if (op_str == "max") {
            const auto t0 = chrono::high_resolution_clock::now();
            auto seq = sequential_max(data);
            const auto t1 = chrono::high_resolution_clock::now();
            chrono::duration<double> seq_time = t1 - t0;

            const auto p0 = chrono::high_resolution_clock::now();
            auto par = parallel_max(data);
            const auto p1 = chrono::high_resolution_clock::now();
            chrono::duration<double> par_time = p1 - p0;

            double abs_err = fabs((double)seq - (double)par);
            double rel_err = fabs(seq) > 0 ? abs_err / fabs((double)seq) : 0.0;

            cout << "[SEQ] Max: " << seq << " time(ms): " << seq_time.count()*1000 << endl;
            cout << "[PAR] Max: " << par << " time(ms): " << par_time.count()*1000 << endl;
            cout << "[CONFIG] threads=" << threads << " chunk=" << chunk << endl;

            // string filename = "../results/openmp/openmp_max_int.csv";
            string filename = "../results/openmp/chunks/openmp_max_int.csv";

            ofstream fout(filename, ios::app);
            fout << power << "," << threads << "," << par_time.count()*1000 << "," << abs_err << "," << rel_err << "," << chunk << endl;
            fout.close();
        } else if (op_str == "prefix") {
            const auto t0 = chrono::high_resolution_clock::now();
            auto seq_scan = sequential_exclusive_scan(data);
            const auto t1 = chrono::high_resolution_clock::now();
            chrono::duration<double> seq_time = t1 - t0;

            vector<int64_t> par_data = data;
            const auto p0 = chrono::high_resolution_clock::now();
            blelloch_scan(par_data);
            const auto p1 = chrono::high_resolution_clock::now();
            chrono::duration<double> par_time = p1 - p0;

            double err_sq = 0.0, denom_sq = 0.0;
            for (size_t i = 0; i < seq_scan.size(); ++i) {
                double d = (double)seq_scan[i] - (double)par_data[i];
                err_sq += d*d;
                denom_sq += ((double)seq_scan[i])*((double)seq_scan[i]);
            }
            double abs_err = sqrt(err_sq);
            double denom = sqrt(denom_sq);
            double rel_err = denom > 0 ? abs_err / denom : 0.0;

            cout << "[SEQ] Prefix time(ms): " << seq_time.count()*1000 << endl;
            cout << "[PAR] Prefix time(ms): " << par_time.count()*1000 << endl;
            cout << "[CONFIG] threads=" << threads << " chunk=" << chunk << endl;

            // string filename = "../results/openmp/openmp_prefix_int.csv";
            string filename = "../results/openmp/chunks/openmp_prefix_int.csv";

            ofstream fout(filename, ios::app);
            fout << power << "," << threads << "," << par_time.count()*1000 << "," << abs_err << "," << rel_err << "," << chunk << endl;
            fout.close();
        } else {
            cerr << "Unknown operation: " << op_str << endl;
            return 1;
        }
    } else if (type_str == "double") {
        vector<double> data(N);
        mt19937_64 gen(1234);
        uniform_real_distribution<double> dist(0.0, 1000000.0);
        for (auto &x : data) x = dist(gen);

        if (op_str == "sum") {
            const auto t0 = chrono::high_resolution_clock::now();
            auto seq = sequential_sum(data);
            const auto t1 = chrono::high_resolution_clock::now();
            chrono::duration<double> seq_time = t1 - t0;

            const auto p0 = chrono::high_resolution_clock::now();
            auto par = parallel_sum(data);
            const auto p1 = chrono::high_resolution_clock::now();
            chrono::duration<double> par_time = p1 - p0;

            double abs_err = fabs(seq - par);
            double rel_err = fabs(seq) > 0 ? abs_err / fabs(seq) : 0.0;

            cout << "[SEQ] Sum: " << seq << " time(ms): " << seq_time.count()*1000 << endl;
            cout << "[PAR] Sum: " << par << " time(ms): " << par_time.count()*1000 << endl;
            cout << "[CONFIG] threads=" << threads << " chunk=" << chunk << endl;

            // string filename = "../results/openmp/openmp_sum_double.csv";
            string filename = "../results/openmp/chunks/openmp_sum_double.csv";

            ofstream fout(filename, ios::app);
            fout << power << "," << threads << "," << par_time.count()*1000 << "," << abs_err << "," << rel_err << "," << chunk << endl;
            fout.close();
        } else if (op_str == "min") {
            const auto t0 = chrono::high_resolution_clock::now();
            auto seq = sequential_min(data);
            const auto t1 = chrono::high_resolution_clock::now();
            chrono::duration<double> seq_time = t1 - t0;

            const auto p0 = chrono::high_resolution_clock::now();
            auto par = parallel_min(data);
            const auto p1 = chrono::high_resolution_clock::now();
            chrono::duration<double> par_time = p1 - p0;

            double abs_err = fabs(seq - par);
            double rel_err = fabs(seq) > 0 ? abs_err / fabs(seq) : 0.0;

            cout << "[SEQ] Min: " << seq << " time(ms): " << seq_time.count()*1000 << endl;
            cout << "[PAR] Min: " << par << " time(ms): " << par_time.count()*1000 << endl;
            cout << "[CONFIG] threads=" << threads << " chunk=" << chunk << endl;

            // string filename = "../results/openmp/openmp_min_double.csv";
            string filename = "../results/openmp/chunks/openmp_min_double.csv";

            ofstream fout(filename, ios::app);
            fout << power << "," << threads << "," << par_time.count()*1000 << "," << abs_err << "," << rel_err << "," << chunk << endl;
            fout.close();
        } else if (op_str == "max") {
            const auto t0 = chrono::high_resolution_clock::now();
            auto seq = sequential_max(data);
            const auto t1 = chrono::high_resolution_clock::now();
            chrono::duration<double> seq_time = t1 - t0;

            const auto p0 = chrono::high_resolution_clock::now();
            auto par = parallel_max(data);
            const auto p1 = chrono::high_resolution_clock::now();
            chrono::duration<double> par_time = p1 - p0;

            double abs_err = fabs(seq - par);
            double rel_err = fabs(seq) > 0 ? abs_err / fabs(seq) : 0.0;

            cout << "[SEQ] Max: " << seq << " time(ms): " << seq_time.count()*1000 << endl;
            cout << "[PAR] Max: " << par << " time(ms): " << par_time.count()*1000 << endl;
            cout << "[CONFIG] threads=" << threads << " chunk=" << chunk << endl;

            // string filename = "../results/openmp/openmp_max_double.csv";
            string filename = "../results/openmp/chunks/openmp_max_double.csv";

            ofstream fout(filename, ios::app);
            fout << power << "," << threads << "," << par_time.count()*1000 << "," << abs_err << "," << rel_err << "," << chunk << endl;
            fout.close();
        } else if (op_str == "prefix") {
            const auto t0 = chrono::high_resolution_clock::now();
            auto seq_scan = sequential_exclusive_scan(data);
            const auto t1 = chrono::high_resolution_clock::now();
            chrono::duration<double> seq_time = t1 - t0;

            vector<double> par_data = data;
            const auto p0 = chrono::high_resolution_clock::now();
            blelloch_scan(par_data);
            const auto p1 = chrono::high_resolution_clock::now();
            chrono::duration<double> par_time = p1 - p0;

            double err_sq = 0.0, denom_sq = 0.0;
            for (size_t i = 0; i < seq_scan.size(); ++i) {
                double d = seq_scan[i] - par_data[i];
                err_sq += d*d;
                denom_sq += seq_scan[i]*seq_scan[i];
            }
            double abs_err = sqrt(err_sq);
            double denom = sqrt(denom_sq);
            double rel_err = denom > 0 ? abs_err / denom : 0.0;

            cout << "[SEQ] Prefix time(ms): " << seq_time.count()*1000 << endl;
            cout << "[PAR] Prefix time(ms): " << par_time.count()*1000 << endl;
            cout << "[CONFIG] threads=" << threads << " chunk=" << chunk << endl;

            // string filename = "../results/openmp/openmp_prefix_double.csv";
            string filename = "../results/openmp/chunks/openmp_prefix_double.csv";

            ofstream fout(filename, ios::app);
            fout << power << "," << threads << "," << par_time.count()*1000 << "," << abs_err << "," << rel_err << "," << chunk << endl;
            fout.close();
        } else {
            cerr << "Unknown operation: " << op_str << endl;
            return 1;
        }
    } else {
        cerr << "Unknown type: " << type_str << endl;
        return 1;
    }

    return 0;
}