#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <chrono>

using namespace std;

// Sekwencyjne wersje (używane tylko przez rank 0 do weryfikacji)
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

void compute_counts_displs(int N, int size, vector<int>& counts, vector<int>& displs) {
    counts.resize(size);
    displs.resize(size);

    int base = N / size;
    int rest = N % size;
    cout << "Base: " << base << ", Rest: " << rest << endl;

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
void mpi_sum(const vector<T>& full_vec, int rank, int size, const vector<int>& counts, const vector<int>& displs) {
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = chrono::high_resolution_clock::now();

    vector<T> local_vec(counts[rank]);
    MPI_Scatterv(full_vec.data(), counts.data(), displs.data(), MPI_LONG_LONG,
                 local_vec.data(), counts[rank], MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    T local_sum = 0;
    for (auto& x : local_vec) local_sum += x;

    T global_sum = 0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    if (rank == 0) {
        // SUMA - SEKWENCYJNIE
        const auto start_seq_sum = chrono::high_resolution_clock::now();
        auto seq_sum = sequential_sum(full_vec);
        const auto end_seq_sum = chrono::high_resolution_clock::now();
        chrono::duration<double> seq_time_sum = end_seq_sum - start_seq_sum;

        cout << "[SEQ] Sum of all elements is equal: " << seq_sum << endl;
        cout << "[SEQ] Execution time for sum reduction sequential method: " << seq_time_sum.count() * 1000 << "ms" << endl << endl;

        cout << "[MPI] Sum of all elements is equal: " << global_sum << endl;
        cout << "[MPI] Execution time for sum reduction parallel method: " << duration.count() * 1000 << "ms" << endl << endl;
    }
}

// ==========================
// MPI MIN
// ==========================
template <typename T>
void mpi_min(const vector<T>& full_vec, int rank, int size, const vector<int>& counts, const vector<int>& displs) {
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = chrono::high_resolution_clock::now();

    vector<T> local_vec(counts[rank]);
    MPI_Scatterv(full_vec.data(), counts.data(), displs.data(), MPI_LONG_LONG,
                 local_vec.data(), counts[rank], MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    T local_min = numeric_limits<T>::max();
    for (auto& x : local_vec) local_min = min(local_min, x);

    T global_min = numeric_limits<T>::max();
    MPI_Allreduce(&local_min, &global_min, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);

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
    }
}

// ==========================
// MPI MAX
// ==========================
template <typename T>
void mpi_max(const vector<T>& full_vec, int rank, int size, const vector<int>& counts, const vector<int>& displs) {
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = chrono::high_resolution_clock::now();

    vector<T> local_vec(counts[rank]);
    MPI_Scatterv(full_vec.data(), counts.data(), displs.data(), MPI_LONG_LONG,
                 local_vec.data(), counts[rank], MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    T local_max = numeric_limits<T>::min();
    for (auto& x : local_vec) local_max = max(local_max, x);

    T global_max = numeric_limits<T>::min();
    MPI_Allreduce(&local_max, &global_max, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

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
    }
}


// ==========================
// MPI PREFIX-SUM (exclusive scan)
// ==========================
template <typename T>
void mpi_prefix_sum(const vector<T>& full_vec, int local_n, int rank, int size) {
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = chrono::high_resolution_clock::now();

    vector<T> local_data(local_n);
    MPI_Scatter(full_vec.data(), local_n, MPI_LONG_LONG,
                local_data.data(), local_n, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // 1. Lokalne sumy
    T local_sum = std::accumulate(local_data.begin(), local_data.end(), T(0));

    // printf("Rank %d has local sum: %lld\n", rank, static_cast<long long>(local_sum));

    // 2. Każdy proces zna swoją lokalną sumę
    std::vector<T> all_sums(size);
    MPI_Allgather(&local_sum, 1, MPI_LONG_LONG, all_sums.data(), 1, MPI_LONG_LONG, MPI_COMM_WORLD);

    // cout <<"im here: " << rank <<endl;
    // for (int i = 0; i < all_sums.size(); i++) {
    //     cout << "all_sums[" << i << "] = " << all_sums[i] << endl;
    // }

    // 3. Oblicz offset prefiksowy dla każdego procesu
    std::vector<T> prefix_sums(size);
    prefix_sums[0] = 0.0;
    for (int i = 1; i < size; ++i)
        prefix_sums[i] = prefix_sums[i - 1] + all_sums[i - 1];

    T offset = prefix_sums[rank];

    // 4. Wewnątrzprocesowy exclusive scan
    vector<T> local_scan(local_n);
    if (local_n > 0) {
        local_scan[0] = offset;
        for (int i = 1; i < local_n; i++)
            local_scan[i] = local_scan[i - 1] + local_data[i - 1];
    }

    // 5. (opcjonalnie) allgather – pełny wynik globalny
    vector<T> global_scan(size * local_n);
    MPI_Allgather(local_scan.data(), local_n, MPI_LONG_LONG,
                  global_scan.data(), local_n, MPI_LONG_LONG, MPI_COMM_WORLD);
                
    // local_data = std::move(local_scan);
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;

    if (rank == 0) {
        const auto start_seq_scan = chrono::high_resolution_clock::now();
        auto seq_scan = sequential_exclusive_scan(full_vec);
        const auto end_seq_scan = chrono::high_resolution_clock::now();
        chrono::duration<double> seq_time_scan = end_seq_scan - start_seq_scan;

        cout << "[SEQ] Execution time for exclusive prefix-sum: " << seq_time_scan.count() * 1000 << " ms" << endl;
        cout << "[MPI] Execution time for exclusive prefix-sum: " << duration.count() * 1000 << " ms" << endl;

        for (size_t i = 0; i < 10 && i < global_scan.size(); i++) {
            cout << "[MPI] global_scan[" << i << "] = " << global_scan[i]
                 << ", [SEQ] seq_scan[" << i << "] = " << seq_scan[i] << endl;
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const size_t N = 1 << 25;

    vector<int64_t> data;
    if (rank == 0) {
        data.resize(N);
        mt19937_64 gen(1234);
        uniform_int_distribution<int64_t> dist(0, 1000000);
        for (auto& x : data) x = dist(gen);
        cout << "Wektor o rozmiarze " << N << " elementów wygenerowany.\n\n";

        // for (size_t i = 0; i < min(size_t(10), data.size()); i++) {
        //     cout << "data[" << i << "] = " << data[i] << endl;
        // }
    }

    vector<int> counts, displs;
    compute_counts_displs(N, size, counts, displs);

    int local_n = N / size;

    // mpi_sum<int64_t>(data, rank, size, counts, displs);
    // mpi_min<int64_t>(data, rank, size, counts, displs);
    // mpi_max<int64_t>(data, rank, size, counts, displs);
    mpi_prefix_sum<int64_t>(data, local_n, rank, size);

    MPI_Finalize();
    return 0;
}
