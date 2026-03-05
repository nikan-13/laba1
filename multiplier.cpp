#include <iostream>
#include <vector>
#include <fstream>
#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <iomanip>

void fill_matrix(std::vector<double>& mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = (double)(rand() % 100) / 10.0;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) std::cerr << "Usage: " << argv[0] << " <matrix_size>" << std::endl;
        MPI_Finalize();
        return 1;
    }

    int N = std::atoi(argv[1]);
    
    if (N % size != 0) {
        if (rank == 0) std::cerr << "Error: Matrix size must be divisible by number of processes." << std::endl;
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = N / size;

    std::vector<double> A, B, C;
    std::vector<double> local_A(rows_per_proc * N);
    std::vector<double> local_C(rows_per_proc * N);

    double start_time, end_time;

    if (rank == 0) {
        A.resize(N * N);
        B.resize(N * N);
        C.resize(N * N);
        
        srand(time(NULL)); // Инициализация ГСЧ
        fill_matrix(A, N, N);
        fill_matrix(B, N, N);

        start_time = MPI_Wtime();
    }

    // Рассылка A
    MPI_Scatter(A.data(), rows_per_proc * N, MPI_DOUBLE, 
                local_A.data(), rows_per_proc * N, MPI_DOUBLE, 
                0, MPI_COMM_WORLD);

    // Трансляция B
    if (rank != 0) {
        B.resize(N * N);
    }
    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Вычисления с volatile, чтобы компилятор не выкинул циклы
    for (int i = 0; i < rows_per_proc; ++i) {
        for (int j = 0; j < N; ++j) {
            volatile double sum = 0.0; // volatile предотвращает оптимизацию
            for (int k = 0; k < N; ++k) {
                sum += local_A[i * N + k] * B[k * N + j];
            }
            local_C[i * N + j] = sum;
        }
    }

    // Сбор C
    MPI_Gather(local_C.data(), rows_per_proc * N, MPI_DOUBLE,
               C.data(), rows_per_proc * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        end_time = MPI_Wtime();
        double elapsed = end_time - start_time;

        // Вывод в формате для парсинга Python
        std::cout << "Time: " << elapsed << " Size: " << N << std::endl;

        // Запись в файл (опционально, для верификации)
        std::ofstream outfile("result_matrix.txt");
        if (outfile.is_open()) {
            outfile << N << "\n"; 
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    outfile << C[i * N + j] << " ";
                }
                outfile << "\n";
            }
            outfile.close();
        }
    }

    MPI_Finalize();
    return 0;
}