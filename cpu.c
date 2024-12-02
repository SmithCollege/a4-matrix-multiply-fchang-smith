#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

double get_clock() {
    struct timeval tv;
    int ok = gettimeofday(&tv, NULL);
    if (ok < 0) {
        printf("gettimeofday error\n");
        return -1;
    }
    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}


void cpuMatrixMultiply(double* A, double* B, double* C, int N) {
    int i, j, k;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            double sum = 0.0;
            for (k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main() {
    int N = 1024; // Matrix size
    double* A = (double*)malloc(N * N * sizeof(double));
    double* B = (double*)malloc(N * N * sizeof(double));
    double* C = (double*)malloc(N * N * sizeof(double));

    if (A == NULL || B == NULL || C == NULL) {
        printf("Memory allocation failed\n");
        return -1;
    }

    // Initialize matrices
    int i;
    for (i = 0; i < N * N; ++i) {
        A[i] = 1.0;
        B[i] = 1.0;
    }

    double start = get_clock();
    cpuMatrixMultiply(A, B, C, N);
    double end = get_clock();

    printf("Matrix Size: %d x %d\n", N, N);
    printf("CPU Time: %f ms\n", (end - start) * 1000);

    free(A);
    free(B);
    free(C);
    return 0;
}
