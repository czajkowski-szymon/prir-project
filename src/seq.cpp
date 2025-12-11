#include <iostream>
#include <algorithm>
#include <limits>
#include <vector>
#include <random>
#include <chrono>
#include "sequential.hpp"
#include <fstream>
#include <numeric>
#include <cmath>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <type: int|double> <operation: sum|min|max|prefix> [power]" << endl;
        return 1;
    }
    string type_str = argv[1] ? argv[1] : "double";
    string op_str = argv[2] ? argv[2] : "sum";
    string power = argv[3] ? argv[3] : "20";

    const size_t N = 1 << stoi(power);

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

            cout << "[SEQ] Sum: " << seq << " time(ms): " << seq_time.count()*1000 << endl;

            string filename = "../results/seq/seq_sum_int.csv";
            ofstream fout(filename, ios::app);
            fout << power << "," << seq_time.count()*1000 << endl;
            fout.close();
        } else if (op_str == "min") {
            const auto t0 = chrono::high_resolution_clock::now();
            auto seq = sequential_min(data);
            const auto t1 = chrono::high_resolution_clock::now();
            chrono::duration<double> seq_time = t1 - t0;

            cout << "[SEQ] Min: " << seq << " time(ms): " << seq_time.count()*1000 << endl;

            string filename = "../results/seq/seq_min_int.csv";
            ofstream fout(filename, ios::app);
            fout << power << "," << seq_time.count()*1000 << endl;
            fout.close();
        } else if (op_str == "max") {
            const auto t0 = chrono::high_resolution_clock::now();
            auto seq = sequential_max(data);
            const auto t1 = chrono::high_resolution_clock::now();
            chrono::duration<double> seq_time = t1 - t0;

            cout << "[SEQ] Max: " << seq << " time(ms): " << seq_time.count()*1000 << endl;

            string filename = "../results/seq/seq_max_int.csv";
            ofstream fout(filename, ios::app);
            fout << power << "," << seq_time.count()*1000 << endl;
            fout.close();
        } else if (op_str == "prefix") {
            const auto t0 = chrono::high_resolution_clock::now();
            auto seq_scan = sequential_exclusive_scan(data);
            const auto t1 = chrono::high_resolution_clock::now();
            chrono::duration<double> seq_time = t1 - t0;

            cout << "[SEQ] Prefix time(ms): " << seq_time.count()*1000 << endl;

            string filename = "../results/seq/seq_prefix_int.csv";
            ofstream fout(filename, ios::app);
            fout << power << "," << seq_time.count()*1000 << endl;
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

            cout << "[SEQ] Sum: " << seq << " time(ms): " << seq_time.count()*1000 << endl;

            string filename = "../results/seq/seq_sum_double.csv";
            ofstream fout(filename, ios::app);
            fout << power << "," << seq_time.count()*1000 << endl;
            fout.close();
        } else if (op_str == "min") {
            const auto t0 = chrono::high_resolution_clock::now();
            auto seq = sequential_min(data);
            const auto t1 = chrono::high_resolution_clock::now();
            chrono::duration<double> seq_time = t1 - t0;

            cout << "[SEQ] Min: " << seq << " time(ms): " << seq_time.count()*1000 << endl;

            string filename = "../results/seq/seq_min_double.csv";
            ofstream fout(filename, ios::app);
            fout << power << "," << seq_time.count()*1000 << endl;
            fout.close();
        } else if (op_str == "max") {
            const auto t0 = chrono::high_resolution_clock::now();
            auto seq = sequential_max(data);
            const auto t1 = chrono::high_resolution_clock::now();
            chrono::duration<double> seq_time = t1 - t0;

            cout << "[SEQ] Max: " << seq << " time(ms): " << seq_time.count()*1000 << endl;

            string filename = "../results/seq/seq_max_double.csv";
            ofstream fout(filename, ios::app);
            fout << power << "," << seq_time.count()*1000 << endl;
            fout.close();
        } else if (op_str == "prefix") {
            const auto t0 = chrono::high_resolution_clock::now();
            auto seq_scan = sequential_exclusive_scan(data);
            const auto t1 = chrono::high_resolution_clock::now();
            chrono::duration<double> seq_time = t1 - t0;

            cout << "[SEQ] Prefix time(ms): " << seq_time.count()*1000 << endl;

            string filename = "../results/seq/seq_prefix_double.csv";
            ofstream fout(filename, ios::app);
            fout << power << "," << seq_time.count()*1000 << endl;
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