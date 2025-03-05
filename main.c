/*
 * Program: Latin Square Matrix Normalizer with Strassen, OpenMP, and AVX2
 * Description:
 *   - Reads a vector of double values from the user.
 *   - Constructs a Latin square matrix M by cyclically shifting the input vector.
 *   - Pads M (if needed) to a power-of-2 size, and computes the product P = M * M
 *     using Strassen matrix multiplication.
 *   - Strassen multiplication uses OpenMP tasks for parallelism and AVX2 intrinsics
 *     for the base-case multiplication.
 *   - After multiplication, checks for division by 0.0 and normalizes P using the
 *     affine mapping:
 *         f(x) = (x - minP) / (maxP - minP)
 *     where minP and maxP are the minimum and maximum entries in P.
 *   - The program loops indefinitely (use Ctrl+C to exit).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <time.h>

// Function to compute the next power of 2 greater than or equal to n.
int nextPowerOf2(int n) {
    int power = 1;
    while (power < n)
        power *= 2;
    return power;
}

// Allocate an n x n matrix (initialized to 0).
double** allocateMatrix(int n) {
    double **mat = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        mat[i] = (double *)calloc(n, sizeof(double));
    }
    return mat;
}

// Free an n x n matrix.
void freeMatrix(double **mat, int n) {
    for (int i = 0; i < n; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Matrix addition: C = A + B.
void addMatrix(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
}

// Matrix subtraction: C = A - B.
void subtractMatrix(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
}

// Base multiplication using AVX2 intrinsics (for small matrices).
void baseMultiply(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            __m256d sumVec = _mm256_setzero_pd();
            int k;
            for (k = 0; k <= n - 4; k += 4) {
                __m256d vecA = _mm256_loadu_pd(&A[i][k]);
                __m256d vecB = _mm256_set_pd(B[k+3][j], B[k+2][j], B[k+1][j], B[k][j]);
                __m256d prod = _mm256_mul_pd(vecA, vecB);
                sumVec = _mm256_add_pd(sumVec, prod);
            }
            double temp[4];
            _mm256_storeu_pd(temp, sumVec);
            double sum = temp[0] + temp[1] + temp[2] + temp[3];
            for (; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

// Recursive Strassen multiplication.
// Assumes matrices A, B, and C are all n x n, with n a power of 2.
void strassenMultiply(double **A, double **B, double **C, int n) {
    // Threshold for switching to base multiplication.
    if(n <= 64) {
        baseMultiply(A, B, C, n);
        return;
    }
    int newSize = n / 2;
    // Allocate submatrices for A and B partitions.
    double **A11 = allocateMatrix(newSize);
    double **A12 = allocateMatrix(newSize);
    double **A21 = allocateMatrix(newSize);
    double **A22 = allocateMatrix(newSize);
    double **B11 = allocateMatrix(newSize);
    double **B12 = allocateMatrix(newSize);
    double **B21 = allocateMatrix(newSize);
    double **B22 = allocateMatrix(newSize);

    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j+newSize];
            A21[i][j] = A[i+newSize][j];
            A22[i][j] = A[i+newSize][j+newSize];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j+newSize];
            B21[i][j] = B[i+newSize][j];
            B22[i][j] = B[i+newSize][j+newSize];
        }
    }

    // Allocate matrices for the 7 Strassen products.
    double **M1 = allocateMatrix(newSize);
    double **M2 = allocateMatrix(newSize);
    double **M3 = allocateMatrix(newSize);
    double **M4 = allocateMatrix(newSize);
    double **M5 = allocateMatrix(newSize);
    double **M6 = allocateMatrix(newSize);
    double **M7 = allocateMatrix(newSize);

    // Temporary matrices for sums/differences.
    double **T1 = allocateMatrix(newSize);
    double **T2 = allocateMatrix(newSize);

    // Use OpenMP tasks for parallelism.
    #pragma omp task shared(M1, T1, T2) if(newSize > 64)
    {
        addMatrix(A11, A22, T1, newSize);      // T1 = A11 + A22
        addMatrix(B11, B22, T2, newSize);        // T2 = B11 + B22
        strassenMultiply(T1, T2, M1, newSize);
    }

    #pragma omp task shared(M2, T1) if(newSize > 64)
    {
        addMatrix(A21, A22, T1, newSize);      // T1 = A21 + A22
        strassenMultiply(T1, B11, M2, newSize);
    }

    #pragma omp task shared(M3, T2) if(newSize > 64)
    {
        subtractMatrix(B12, B22, T2, newSize); // T2 = B12 - B22
        strassenMultiply(A11, T2, M3, newSize);
    }

    #pragma omp task shared(M4, T2) if(newSize > 64)
    {
        subtractMatrix(B21, B11, T2, newSize); // T2 = B21 - B11
        strassenMultiply(A22, T2, M4, newSize);
    }

    #pragma omp task shared(M5, T1) if(newSize > 64)
    {
        addMatrix(A11, A12, T1, newSize);      // T1 = A11 + A12
        strassenMultiply(T1, B22, M5, newSize);
    }

    #pragma omp task shared(M6, T1, T2) if(newSize > 64)
    {
        subtractMatrix(A21, A11, T1, newSize); // T1 = A21 - A11
        addMatrix(B11, B12, T2, newSize);      // T2 = B11 + B12
        strassenMultiply(T1, T2, M6, newSize);
    }

    #pragma omp task shared(M7, T1, T2) if(newSize > 64)
    {
        subtractMatrix(A12, A22, T1, newSize); // T1 = A12 - A22
        addMatrix(B21, B22, T2, newSize);      // T2 = B21 + B22
        strassenMultiply(T1, T2, M7, newSize);
    }

    #pragma omp taskwait

    // Allocate submatrices for result.
    double **C11 = allocateMatrix(newSize);
    double **C12 = allocateMatrix(newSize);
    double **C21 = allocateMatrix(newSize);
    double **C22 = allocateMatrix(newSize);

    // Compute C11 = M1 + M4 - M5 + M7.
    addMatrix(M1, M4, T1, newSize);       // T1 = M1 + M4
    subtractMatrix(T1, M5, T1, newSize);   // T1 = T1 - M5
    addMatrix(T1, M7, C11, newSize);       // C11 = T1 + M7

    // Compute C12 = M3 + M5.
    addMatrix(M3, M5, C12, newSize);

    // Compute C21 = M2 + M4.
    addMatrix(M2, M4, C21, newSize);

    // Compute C22 = M1 - M2 + M3 + M6.
    subtractMatrix(M1, M2, T1, newSize);  // T1 = M1 - M2
    addMatrix(T1, M3, T1, newSize);         // T1 = T1 + M3
    addMatrix(T1, M6, C22, newSize);         // C22 = T1 + M6

    // Combine submatrices into result matrix C.
    for (int i = 0; i < newSize; i++) {
        for (int j = 0; j < newSize; j++) {
            C[i][j] = C11[i][j];
            C[i][j+newSize] = C12[i][j];
            C[i+newSize][j] = C21[i][j];
            C[i+newSize][j+newSize] = C22[i][j];
        }
    }

    // Free all temporary matrices.
    freeMatrix(A11, newSize); freeMatrix(A12, newSize);
    freeMatrix(A21, newSize); freeMatrix(A22, newSize);
    freeMatrix(B11, newSize); freeMatrix(B12, newSize);
    freeMatrix(B21, newSize); freeMatrix(B22, newSize);
    freeMatrix(M1, newSize); freeMatrix(M2, newSize);
    freeMatrix(M3, newSize); freeMatrix(M4, newSize);
    freeMatrix(M5, newSize); freeMatrix(M6, newSize);
    freeMatrix(M7, newSize);
    freeMatrix(T1, newSize); freeMatrix(T2, newSize);
    freeMatrix(C11, newSize); freeMatrix(C12, newSize);
    freeMatrix(C21, newSize); freeMatrix(C22, newSize);
}

// Pad an n x n matrix into an m x m matrix (m >= n).
double** padMatrix(double **mat, int n, int m) {
    double **pad = allocateMatrix(m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            pad[i][j] = mat[i][j];
    return pad;
}

// Unpad an m x m matrix into an n x n matrix.
void unpadMatrix(double **pad, double **mat, int n, int m) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            mat[i][j] = pad[i][j];
}

int main(void) {
    // Seed the random number generator.
    srand(time(NULL));

    int mode;
    printf("Choose mode:\n");
    printf("  (1) Chain Mode (random initial vector with normalized first row fed as next input)\n");
    printf("  (2) Manual Input Mode\n");
    printf("Enter mode (1 or 2): ");
    if (scanf("%d", &mode) != 1) {
        printf("Invalid input.\n");
        return 1;
    }

    if (mode == 1) {
        int n, iterations;
        printf("Enter the size of the vector: ");
        if (scanf("%d", &n) != 1 || n <= 0) {
            printf("Invalid size.\n");
            return 1;
        }
        printf("Enter the number of chain iterations (enter 0 for infinite loop): ");
        if (scanf("%d", &iterations) != 1) {
            printf("Invalid input.\n");
            return 1;
        }

        // Generate the initial random vector with values in [-10, 10].
        double *vector = (double *)malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) {
            double r = ((double)rand() / RAND_MAX) * 20.0 - 10.0;
            vector[i] = r;
        }

        int iter = 0;
        while (iterations == 0 || iter < iterations) {
            printf("\n=== Chain Iteration %d ===\n", iter + 1);
            printf("Input Vector:\n");
            for (int i = 0; i < n; i++) {
                printf("%.2lf ", vector[i]);
            }
            printf("\n");

            // Construct the Latin square matrix M from the vector.
            double **M = allocateMatrix(n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    M[i][j] = vector[(j + i) % n];

            // Determine padded size (power of 2) and pad M if necessary.
            int m = nextPowerOf2(n);
            double **M_pad = (m == n) ? M : padMatrix(M, n, m);
            double **P_pad = allocateMatrix(m);

            // Compute P = M * M using Strassen multiplication.
            #pragma omp parallel
            {
                #pragma omp single
                {
                    strassenMultiply(M_pad, M_pad, P_pad, m);
                }
            }

            // Unpad product matrix P_pad into P (n x n).
            double **P = allocateMatrix(n);
            if (m != n) {
                unpadMatrix(P_pad, P, n, m);
            } else {
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++)
                        P[i][j] = P_pad[i][j];
            }

            // Compute the minimum and maximum entries in P.
            double minP = P[0][0], maxP = P[0][0];
            for (int i = 0; i < n; i++){
                for (int j = 0; j < n; j++){
                    if (P[i][j] < minP)
                        minP = P[i][j];
                    if (P[i][j] > maxP)
                        maxP = P[i][j];
                }
            }

            // Print Matrix M.
            printf("\nMatrix M (Latin Square):\n");
            for (int i = 0; i < n; i++){
                for (int j = 0; j < n; j++){
                    printf("%.2lf ", M[i][j]);
                }
                printf("\n");
            }

            // Normalize and print Matrix P using the affine mapping.
            if (fabs(maxP - minP) < 1e-12) {
                printf("\nNormalization skipped: max and min are equal (division by zero).\n");
            } else {
                printf("\nNormalized Matrix P (affine mapping: (P_{i,j} - %.2lf) / (%.2lf - %.2lf)):\n", minP, maxP, minP);
                for (int i = 0; i < n; i++){
                    for (int j = 0; j < n; j++){
                        P[i][j] = (P[i][j] - minP) / (maxP - minP);
                        printf("%.2lf ", P[i][j]);
                    }
                    printf("\n");
                }
            }

            // If normalization was performed, check if the first row remains unchanged.
            if (fabs(maxP - minP) >= 1e-12) {
                int same = 1;
                double epsilon = 1e-6;
                for (int j = 0; j < n; j++) {
                    if (fabs(vector[j] - P[0][j]) > epsilon) {
                        same = 0;
                        break;
                    }
                }
                if (same) {
                    printf("\nNormalized first row remained unchanged. Generating new random input vector.\n");
                    for (int j = 0; j < n; j++) {
                        double r = ((double)rand() / RAND_MAX) * 20.0 - 10.0;
                        vector[j] = r;
                    }
                } else {
                    // Update the input vector with the first row of the normalized matrix.
                    for (int j = 0; j < n; j++) {
                        vector[j] = P[0][j];
                    }
                }
            } else {
                printf("\nInput vector remains unchanged due to division by zero.\n");
            }

            // Free allocated memory.
            freeMatrix(M, n);
            if (m != n) {
                freeMatrix(M_pad, m);
            }
            freeMatrix(P_pad, m);
            freeMatrix(P, n);

            iter++;
        }
        free(vector);
    } else {
        // Manual input mode (loop indefinitely).
        while (1) {
            int n, i, j;
            printf("\nEnter the size of the vector: ");
            if (scanf("%d", &n) != 1 || n <= 0) {
                printf("Invalid size. Please enter a positive integer.\n");
                while(getchar() != '\n');
                continue;
            }
            double *vector = (double *)malloc(n * sizeof(double));
            if (!vector) {
                printf("Memory allocation failed.\n");
                exit(1);
            }
            printf("Enter %d vector element(s): ", n);
            for (i = 0; i < n; i++) {
                if (scanf("%lf", &vector[i]) != 1) {
                    printf("Invalid input. Restarting iteration.\n");
                    while(getchar() != '\n');
                    free(vector);
                    vector = NULL;
                    break;
                }
            }
            if (!vector)
                continue;

            double **M = allocateMatrix(n);
            for (i = 0; i < n; i++)
                for (j = 0; j < n; j++)
                    M[i][j] = vector[(j + i) % n];

            int m = nextPowerOf2(n);
            double **M_pad = (m == n) ? M : padMatrix(M, n, m);
            double **P_pad = allocateMatrix(m);

            #pragma omp parallel
            {
                #pragma omp single
                {
                    strassenMultiply(M_pad, M_pad, P_pad, m);
                }
            }

            double **P = allocateMatrix(n);
            if(m != n) {
                unpadMatrix(P_pad, P, n, m);
            } else {
                for (i = 0; i < n; i++)
                    for (j = 0; j < n; j++)
                        P[i][j] = P_pad[i][j];
            }

            // Compute the minimum and maximum entries in P.
            double minP = P[0][0], maxP = P[0][0];
            for (i = 0; i < n; i++){
                for (j = 0; j < n; j++){
                    if (P[i][j] < minP)
                        minP = P[i][j];
                    if (P[i][j] > maxP)
                        maxP = P[i][j];
                }
            }

            printf("\nMatrix M (Latin Square):\n");
            for (i = 0; i < n; i++){
                for (j = 0; j < n; j++){
                    printf("%.2lf ", M[i][j]);
                }
                printf("\n");
            }

            printf("\nMatrix P (M * M):\n");
            for (i = 0; i < n; i++){
                for (j = 0; j < n; j++){
                    printf("%.2lf ", P[i][j]);
                }
                printf("\n");
            }

            // Normalize and print Matrix P using the affine mapping.
            if (fabs(maxP - minP) < 1e-12) {
                printf("\nNormalization skipped: max and min are equal (division by zero).\n");
            } else {
                printf("\nNormalized Matrix P (affine mapping: (P_{i,j} - %.2lf) / (%.2lf - %.2lf)):\n", minP, maxP, minP);
                for (i = 0; i < n; i++){
                    for (j = 0; j < n; j++){
                        printf("%.2lf ", (P[i][j] - minP) / (maxP - minP));
                    }
                    printf("\n");
                }
            }

            free(vector);
            freeMatrix(M, n);
            if(m != n) {
                freeMatrix(M_pad, m);
            }
            freeMatrix(P_pad, m);
            freeMatrix(P, n);

            printf("\n--------------------------------\n\n");
        }
    }
    return 0;
}
