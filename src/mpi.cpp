#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <chrono>
#include <fstream>
#include <cmath>
#include <string>
#include "sequential.hpp"

using namespace std;

void compute_counts_displs(int N, int size, vector<int>& counts, vector<int>& displs) {
    counts.resize(size);
    displs.resize(size);

    int base = N / size;
    int rest = N % size;

    for (int i = 0; i < size; i++)
        counts[i] = base + (i < rest ? 1 : 0);

    displs[0] = 0;
    for (int i = 1; i < size; i++)
        displs[i] = displs[i - 1] + counts[i - 1];
}

// ==========================
// MPI SUMA
// ==========================
template <typename T>
void mpi_sum(const vector<T>& full_vec, int rank, int size, const vector<int>& counts, const vector<int>& displs, MPI_Datatype datatype, const std::string& type_str, const std::string& power) {
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = chrono::high_resolution_clock::now();

        vector<T> local_vec(counts[rank]);
        MPI_Scatterv(full_vec.data(), counts.data(), displs.data(), datatype,
                    local_vec.data(), counts[rank], datatype, 0, MPI_COMM_WORLD);

        T local_sum = 0;
        for (auto& x : local_vec) local_sum += x;

        T global_sum = 0;
        MPI_Allreduce(&local_sum, &global_sum, 1, datatype, MPI_SUM, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;

    if (rank == 0) {
        const auto start_seq_sum = chrono::high_resolution_clock::now();
        auto seq_sum = sequential_sum(full_vec);
        const auto end_seq_sum = chrono::high_resolution_clock::now();
        chrono::duration<double> seq_time_sum = end_seq_sum - start_seq_sum;

        cout << "[SEQ] Sum of all elements is equal: " << seq_sum << endl;
        cout << "[SEQ] Execution time for sum reduction sequential method: " << seq_time_sum.count() * 1000 << "ms" << endl << endl;

        cout << "[MPI] Sum of all elements is equal: " << global_sum << endl;
        cout << "[MPI] Execution time for sum reduction parallel method: " << duration.count() * 1000 << "ms" << endl << endl;

        double abs_err = fabs(seq_sum - global_sum);
        double rel_err = fabs(seq_sum) > 0 ? abs_err / fabs(seq_sum) : 0.0;
        cout << "[MPI] Absolute error: " << abs_err << endl;
        cout << "[MPI] Relative error: " << rel_err << endl << endl;

        // --- Zapis do pliku ---
        std::string filename = "../results/mpi/mpi_sum_" + type_str + ".csv";
        std::ofstream fout(filename, std::ios::app);
        fout << power << "," << size << "," << duration.count() * 1000 << "," << abs_err << "," << rel_err << std::endl;
        fout.close();
    }
}

// ==========================
// MPI MIN
// ==========================
template <typename T>
void mpi_min(const vector<T>& full_vec, int rank, int size, const vector<int>& counts, const vector<int>& displs, MPI_Datatype datatype, const std::string& type_str, const std::string& power) {
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = chrono::high_resolution_clock::now();

    vector<T> local_vec(counts[rank]);
    MPI_Scatterv(full_vec.data(), counts.data(), displs.data(), datatype,
                 local_vec.data(), counts[rank], datatype, 0, MPI_COMM_WORLD);

    T local_min = numeric_limits<T>::max();
    for (auto& x : local_vec) local_min = min(local_min, x);

    T global_min = numeric_limits<T>::max();
    MPI_Allreduce(&local_min, &global_min, 1, datatype, MPI_MIN, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    if (rank == 0) {
        const auto start_seq_min = chrono::high_resolution_clock::now();
        auto seq_min = sequential_min(full_vec);
        const auto end_seq_min = chrono::high_resolution_clock::now();
        chrono::duration<double> seq_time_min = end_seq_min - start_seq_min;

        cout << "[SEQ] Minumum element of all elements is equal: " << seq_min << endl;
        cout << "[SEQ] Execution time for min reduction sequential method: " << seq_time_min.count() * 1000 << "ms" << endl << endl;

        cout << "[MPI] Minumum element of all elements is equal: " << global_min << endl;
        cout << "[MPI] Execution time for min reduction parallel method: " << duration.count() * 1000 << "ms" << endl << endl;

        double abs_err = fabs(seq_min - global_min);
        double rel_err = fabs(seq_min) > 0 ? abs_err / fabs(seq_min) : 0.0;
        cout << "[MPI] Absolute error: " << abs_err << endl;
        cout << "[MPI] Relative error: " << rel_err << endl << endl;

        std::string filename = "../results/mpi/mpi_min_" + type_str + ".csv";
        std::ofstream fout(filename, std::ios::app);
        fout << power << "," << size << "," << duration.count() * 1000 << "," << abs_err << "," << rel_err << std::endl;
        fout.close();
    }
}

// ==========================
// MPI MAX
// ==========================
template <typename T>
void mpi_max(const vector<T>& full_vec, int rank, int size, const vector<int>& counts, const vector<int>& displs, MPI_Datatype datatype, const std::string& type_str, const std::string& power) {
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = chrono::high_resolution_clock::now();

    vector<T> local_vec(counts[rank]);
    MPI_Scatterv(full_vec.data(), counts.data(), displs.data(), datatype,
                 local_vec.data(), counts[rank], datatype, 0, MPI_COMM_WORLD);

    T local_max = numeric_limits<T>::min();
    for (auto& x : local_vec) local_max = max(local_max, x);

    T global_max = numeric_limits<T>::min();
    MPI_Allreduce(&local_max, &global_max, 1, datatype, MPI_MAX, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    if (rank == 0) {
        const auto start_seq_max = chrono::high_resolution_clock::now();
        auto seq_max = sequential_max(full_vec);
        const auto end_seq_max = chrono::high_resolution_clock::now();
        chrono::duration<double> seq_time_max = end_seq_max - start_seq_max;

        cout << "[SEQ] Maximum element of all elements is equal: " << seq_max << endl;
        cout << "[SEQ] Execution time for max reduction sequential method: " << seq_time_max.count() * 1000 << "ms" << endl << endl;

        cout << "[MPI] Maximum element of all elements is equal: " << global_max << endl;
        cout << "[MPI] Execution time for max reduction parallel method: " << duration.count() * 1000 << "ms" << endl << endl;

        double abs_err = fabs(seq_max - global_max);
        double rel_err = fabs(seq_max) > 0 ? abs_err / fabs(seq_max) : 0.0;
        cout << "[MPI] Absolute error: " << abs_err << endl;
        cout << "[MPI] Relative error: " << rel_err << endl << endl;

        std::string filename = "../results/mpi/mpi_max_" + type_str + ".csv";
        std::ofstream fout(filename, std::ios::app);
        fout << power << "," << size << "," << duration.count() * 1000 << "," << abs_err << "," << rel_err << std::endl;
        fout.close();
    }
}


 // ==========================
 // MPI PREFIX-SUM (exclusive scan)
 // ==========================
 template <typename T>
 void mpi_prefix_sum(const vector<T>& full_vec, int rank, int size, const vector<int>& counts, const vector<int>& displs, MPI_Datatype datatype, const std::string& type_str, const std::string& power) {
     MPI_Barrier(MPI_COMM_WORLD);
     auto start = chrono::high_resolution_clock::now();
 
     int my_n = counts[rank];
 
     vector<T> local_data(my_n);
     MPI_Scatterv(full_vec.data(), counts.data(), displs.data(), datatype,
                  local_data.data(), my_n, datatype, 0, MPI_COMM_WORLD);
 
     // Lokalne sumy
     T local_sum = std::accumulate(local_data.begin(), local_data.end(), T(0));
 
     std::vector<T> all_sums(size);
     MPI_Allgather(&local_sum, 1, datatype, all_sums.data(), 1, datatype, MPI_COMM_WORLD);
 
     // Offset prefiksowy dla każdego procesu
     std::vector<T> prefix_sums(size);
     prefix_sums[0] = T(0);
     for (int i = 1; i < size; ++i)
         prefix_sums[i] = prefix_sums[i - 1] + all_sums[i - 1];
 
     T offset = prefix_sums[rank];
 
     vector<T> local_scan(my_n);
     if (my_n > 0) {
         local_scan[0] = offset;
         for (int i = 1; i < my_n; i++)
             local_scan[i] = local_scan[i - 1] + local_data[i - 1];
     }
 
     int N = 0;
     for (int c : counts) N += c;
     vector<T> global_scan(N);
     MPI_Allgatherv(local_scan.data(), my_n, datatype,
                    global_scan.data(), counts.data(), displs.data(), datatype, MPI_COMM_WORLD);
 
     auto end = chrono::high_resolution_clock::now();
     chrono::duration<double> duration = end - start;
 
     if (rank == 0) {
         const auto start_seq_scan = chrono::high_resolution_clock::now();
         auto seq_scan = sequential_exclusive_scan(full_vec);
         const auto end_seq_scan = chrono::high_resolution_clock::now();
         chrono::duration<double> seq_time_scan = end_seq_scan - start_seq_scan;
 
         cout << "[SEQ] Execution time for exclusive prefix-sum: " << seq_time_scan.count() * 1000 << " ms" << endl;
         cout << "[MPI] Execution time for exclusive prefix-sum: " << duration.count() * 1000 << " ms" << endl;
 
         double abs_err = 0.0, denom = 0.0;
         for (size_t i = 0; i < seq_scan.size(); i++) {
             abs_err += pow(seq_scan[i] - global_scan[i], 2);
             denom += pow(seq_scan[i], 2);
         }
         abs_err = sqrt(abs_err);
         denom = sqrt(denom);
         double rel_err = denom > 0 ? abs_err / denom : 0.0;
         std::string filename = "../results/mpi/mpi_prefix_" + type_str + ".csv";
         std::ofstream fout(filename, std::ios::app);
         fout << power << "," << size << "," << duration.count() * 1000 << "," << abs_err << "," << rel_err << std::endl;
         fout.close();
     }
 }
 
 int main(int argc, char* argv[]) {
     MPI_Init(&argc, &argv);
 
     int rank, size;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &size);
 
     if (argc < 3) {
         if (rank == 0) {
             cerr << "Usage: " << argv[0] << " <type: int|double> <operation: sum|min|max|prefix> [power]" << endl;
         }
         MPI_Finalize();
         return 1;
     }
     string type_str = argv[1] ? argv[1] : "double";
     string op_str = argv[2] ? argv[2] : "sum";
     string power = argv[3] ? argv[3] : "20";
 
     const size_t N = 1 << stoi(power);
 
     if (type_str == "int") {
         vector<int64_t> data;
         if (rank == 0) {
             data.resize(N);
             mt19937_64 gen(1234);
             uniform_int_distribution<int64_t> dist(0, 1000000);
             for (auto& x : data) x = dist(gen);
             cout << "Wektor o rozmiarze " << N << " elementów wygenerowany.\n\n";
         }
 
         vector<int> counts, displs;
         compute_counts_displs(N, size, counts, displs);
         int local_n = N / size;
 
         if (op_str == "sum") {
             mpi_sum<int64_t>(data, rank, size, counts, displs, MPI_LONG_LONG, "int", power);
         } else if (op_str == "min") {
             mpi_min<int64_t>(data, rank, size, counts, displs, MPI_LONG_LONG, "int", power);
         } else if (op_str == "max") {
             mpi_max<int64_t>(data, rank, size, counts, displs, MPI_LONG_LONG, "int", power);
         } else if (op_str == "prefix") {
             mpi_prefix_sum<int64_t>(data, rank, size, counts, displs, MPI_LONG_LONG, "int", power);
         } else if (rank == 0) {
             cerr << "Unknown operation: " << op_str << endl;
         }
     } else if (type_str == "double") {
         vector<double> data;
         if (rank == 0) {
             data.resize(N);
             mt19937_64 gen(1234);
             uniform_real_distribution<double> dist(0.0, 1000000.0);
             for (auto& x : data) x = dist(gen);
             cout << "Wektor o rozmiarze " << N << " elementów wygenerowany.\n\n";
         }
 
         vector<int> counts, displs;
         compute_counts_displs(N, size, counts, displs);
         int local_n = N / size;
 
         if (op_str == "sum") {
             mpi_sum<double>(data, rank, size, counts, displs, MPI_DOUBLE, "double", power);
         } else if (op_str == "min") {
             mpi_min<double>(data, rank, size, counts, displs, MPI_DOUBLE, "double", power);
         } else if (op_str == "max") {
             mpi_max<double>(data, rank, size, counts, displs, MPI_DOUBLE, "double", power);
         } else if (op_str == "prefix") {
             mpi_prefix_sum<double>(data, rank, size, counts, displs, MPI_DOUBLE, "double", power);
         } else if (rank == 0) {
             cerr << "Unknown operation: " << op_str << endl;
         }
     } else {
         if (rank == 0) {
             cerr << "Unknown type: " << type_str << endl;
         }
     }
 
     MPI_Finalize();
     return 0;
}
