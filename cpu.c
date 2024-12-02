#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Function to get current time in seconds
double get_clock() {
    struct timeval tv;
    int ok;
    ok = gettimeofday(&tv, (void*)0);
    if (ok < 0) {
        printf("gettimeofday error");
    }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

// CPU-based matrix multiplication
void cpuMatrixMultiply(double* A, double* B, double* C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    int sizes[] = {100, 200, 500, 1000, 2000};
    for (int idx = 0; idx < 5; idx++) {
        int N = sizes[idx];
        double* A = (double*)malloc(N * N * sizeof(double));
        double* B = (double*)malloc(N * N * sizeof(double));
        double* C = (double*)malloc(N * N * sizeof(double));

        // Initialize matrices
        for (int i = 0; i < N * N; ++i) {
            A[i] = 1.0;
            B[i] = 1.0;
        }

        // Perform matrix multiplication
        double t0 = get_clock();
        cpuMatrixMultiply(A, B, C, N);
        double t1 = get_clock();

        printf("CPU Matrix Multiply - Size %d: %f s\n", N, t1 - t0);

        // Free memory
        free(A);
        free(B);
        free(C);
    }

    return 0;
}
