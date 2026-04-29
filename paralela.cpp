#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <omp.h>

using namespace std;
using namespace chrono;

typedef vector<vector<double>> Matrix;


Matrix random_matrix(int n) {
    Matrix M(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            M[i][j] = rand() % 10;
    return M;
}

void dividir_matriz(const Matrix &A,
                    Matrix &A11, Matrix &A12,
                    Matrix &A21, Matrix &A22, int half) {
    for (int i = 0; i < half; i++)
        for (int j = 0; j < half; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + half];
            A21[i][j] = A[i + half][j];
            A22[i][j] = A[i + half][j + half];
        }
}

void ensamblar_matriz(const Matrix &C11, const Matrix &C12,
                      const Matrix &C21, const Matrix &C22,
                      Matrix &C, int half) {
    for (int i = 0; i < half; i++)
        for (int j = 0; j < half; j++) {
            C[i][j]            = C11[i][j];
            C[i][j + half]     = C12[i][j];
            C[i + half][j]     = C21[i][j];
            C[i + half][j + half] = C22[i][j];
        }
}


Matrix add(const Matrix &A, const Matrix &B, int n) {
    Matrix C(n, vector<double>(n));
    #pragma omp parallel for collapse(2) if(n > 256)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

Matrix sub(const Matrix &A, const Matrix &B, int n) {
    Matrix C(n, vector<double>(n));
    #pragma omp parallel for collapse(2) if(n > 256)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}


void mult_clasica(const Matrix &A, const Matrix &B, Matrix &C, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = 0.0;
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
                C[i][j] += A[i][k] * B[k][j];
}


void mult_bloques_parallel(const Matrix &A, const Matrix &B, Matrix &C, int n, int b) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += b) {
        for (int jj = 0; jj < n; jj += b) {

            // Inicializar bloque de C a 0 (dentro del hilo dueño del bloque)
            for (int i = ii; i < min(ii + b, n); i++)
                for (int j = jj; j < min(jj + b, n); j++)
                    C[i][j] = 0.0;

            // Acumular contribución de todos los bloques k
            for (int kk = 0; kk < n; kk += b)
                for (int i = ii; i < min(ii + b, n); i++)
                    for (int k = kk; k < min(kk + b, n); k++)
                        for (int j = jj; j < min(jj + b, n); j++)
                            C[i][j] += A[i][k] * B[k][j];
        }
    }
}


Matrix strassen_parallel_puro(const Matrix &A, const Matrix &B, int n) {
    if (n <= 64) {
        Matrix C(n, vector<double>(n, 0.0));
        mult_clasica(A, B, C, n);
        return C;
    }

    int half = n / 2;

    Matrix A11(half, vector<double>(half)), A12(half, vector<double>(half)),
           A21(half, vector<double>(half)), A22(half, vector<double>(half));
    Matrix B11(half, vector<double>(half)), B12(half, vector<double>(half)),
           B21(half, vector<double>(half)), B22(half, vector<double>(half));

    dividir_matriz(A, A11, A12, A21, A22, half);
    dividir_matriz(B, B11, B12, B21, B22, half);

  
    Matrix S1 = add(A11, A22, half), S2 = add(B11, B22, half); 
    Matrix S3 = add(A21, A22, half);                            
    Matrix S4 = sub(B12, B22, half);                           
    Matrix S5 = sub(B21, B11, half);                            
    Matrix S6 = add(A11, A12, half);                            
    Matrix S7 = sub(A21, A11, half), S8 = add(B11, B12, half); 
    Matrix S9 = sub(A12, A22, half), S10= add(B21, B22, half); 
    Matrix M1, M2, M3, M4, M5, M6, M7;

    #pragma omp task shared(M1) if(n > 128)
    M1 = strassen_parallel_puro(S1,  S2,  half);

    #pragma omp task shared(M2) if(n > 128)
    M2 = strassen_parallel_puro(S3,  B11, half);

    #pragma omp task shared(M3) if(n > 128)
    M3 = strassen_parallel_puro(A11, S4,  half);

    #pragma omp task shared(M4) if(n > 128)
    M4 = strassen_parallel_puro(A22, S5,  half);

    #pragma omp task shared(M5) if(n > 128)
    M5 = strassen_parallel_puro(S6,  B22, half);

    #pragma omp task shared(M6) if(n > 128)
    M6 = strassen_parallel_puro(S7,  S8,  half);

    #pragma omp task shared(M7) if(n > 128)
    M7 = strassen_parallel_puro(S9,  S10, half);

    #pragma omp taskwait

    Matrix C11 = add(sub(add(M1, M4, half), M5, half), M7, half);
    Matrix C12 = add(M3, M5, half);
    Matrix C21 = add(M2, M4, half);
    Matrix C22 = add(add(sub(M1, M2, half), M3, half), M6, half);

    Matrix C(n, vector<double>(n, 0.0));
    ensamblar_matriz(C11, C12, C21, C22, C, half);
    return C;
}


Matrix strassen_parallel_recursive(const Matrix &A, const Matrix &B, int n) {
    if (n <= 256) {
        Matrix C(n, vector<double>(n, 0.0));
        mult_bloques_parallel(A, B, C, n, 64);
        return C;
    }

    int half = n / 2;

    Matrix A11(half, vector<double>(half)), A12(half, vector<double>(half)),
           A21(half, vector<double>(half)), A22(half, vector<double>(half));
    Matrix B11(half, vector<double>(half)), B12(half, vector<double>(half)),
           B21(half, vector<double>(half)), B22(half, vector<double>(half));

    dividir_matriz(A, A11, A12, A21, A22, half);
    dividir_matriz(B, B11, B12, B21, B22, half);

    Matrix S1 = add(A11, A22, half), S2 = add(B11, B22, half);
    Matrix S3 = add(A21, A22, half);
    Matrix S4 = sub(B12, B22, half);
    Matrix S5 = sub(B21, B11, half);
    Matrix S6 = add(A11, A12, half);
    Matrix S7 = sub(A21, A11, half), S8 = add(B11, B12, half);
    Matrix S9 = sub(A12, A22, half), S10= add(B21, B22, half);

    Matrix M1, M2, M3, M4, M5, M6, M7;

    #pragma omp task shared(M1) if(n > 512)
    M1 = strassen_parallel_recursive(S1,  S2,  half);

    #pragma omp task shared(M2) if(n > 512)
    M2 = strassen_parallel_recursive(S3,  B11, half);

    #pragma omp task shared(M3) if(n > 512)
    M3 = strassen_parallel_recursive(A11, S4,  half);

    #pragma omp task shared(M4) if(n > 512)
    M4 = strassen_parallel_recursive(A22, S5,  half);

    #pragma omp task shared(M5) if(n > 512)
    M5 = strassen_parallel_recursive(S6,  B22, half);

    #pragma omp task shared(M6) if(n > 512)
    M6 = strassen_parallel_recursive(S7,  S8,  half);

    #pragma omp task shared(M7) if(n > 512)
    M7 = strassen_parallel_recursive(S9,  S10, half);

    #pragma omp taskwait

    Matrix C11 = add(sub(add(M1, M4, half), M5, half), M7, half);
    Matrix C12 = add(M3, M5, half);
    Matrix C21 = add(M2, M4, half);
    Matrix C22 = add(add(sub(M1, M2, half), M3, half), M6, half);

    Matrix C(n, vector<double>(n, 0.0));
    ensamblar_matriz(C11, C12, C21, C22, C, half);
    return C;
}

Matrix strassen_parallel(const Matrix &A, const Matrix &B, int n) {
    int m = 1;
    while (m < n) m *= 2;

    if (m == n) {
        Matrix res;
        #pragma omp parallel
        #pragma omp single
        res = strassen_parallel_recursive(A, B, n);
        return res;
    }

    Matrix A_pad(m, vector<double>(m, 0.0));
    Matrix B_pad(m, vector<double>(m, 0.0));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            A_pad[i][j] = A[i][j];
            B_pad[i][j] = B[i][j];
        }

    Matrix C_pad;
    #pragma omp parallel
    #pragma omp single
    C_pad = strassen_parallel_recursive(A_pad, B_pad, m);

    Matrix C(n, vector<double>(n));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = C_pad[i][j];
    return C;
}

Matrix strassen_puro_wrapper(const Matrix &A, const Matrix &B, int n) {
    Matrix res;
    #pragma omp parallel
    #pragma omp single
    res = strassen_parallel_puro(A, B, n);
    return res;
}

int main() {
    const int N_REPS = 3;  
    vector<int> sizes   = {256, 512, 1024};
    vector<int> threads = {1, 2, 4, 8};
    const int b = 64;

    ofstream file("resultados_parte2.csv");
    file << "n,threads,"
         << "bloques_time,strassen_hibrido_time,strassen_puro_time,"
         << "speedup_bloques,eficiencia_bloques,"
         << "speedup_strassen_hibrido,eficiencia_strassen_hibrido,"
         << "speedup_strassen_puro,eficiencia_strassen_puro\n";

    for (int n : sizes) {
        cout << "\n " << n << endl;

        Matrix A = random_matrix(n);
        Matrix B = random_matrix(n);
        Matrix C(n, vector<double>(n, 0.0));

        double T1_b = 0, T1_sh = 0, T1_sp = 0;
        omp_set_num_threads(1);

        for (int r = 0; r < N_REPS; r++) {
            auto t0 = high_resolution_clock::now();
            mult_bloques_parallel(A, B, C, n, b);
            T1_b += duration<double>(high_resolution_clock::now() - t0).count();

            t0 = high_resolution_clock::now();
            strassen_parallel(A, B, n);
            T1_sh += duration<double>(high_resolution_clock::now() - t0).count();

            t0 = high_resolution_clock::now();
            strassen_puro_wrapper(A, B, n);
            T1_sp += duration<double>(high_resolution_clock::now() - t0).count();
        }
        T1_b  /= N_REPS;
        T1_sh /= N_REPS;
        T1_sp /= N_REPS;

        cout << "Baseline (p=1): bloques=" << T1_b
             << "s  hibrido=" << T1_sh
             << "s  puro=" << T1_sp << "s" << endl;

        for (int t : threads) {
            omp_set_num_threads(t);

            double tb = 0, tsh = 0, tsp = 0;

            for (int r = 0; r < N_REPS; r++) {
                auto t0 = high_resolution_clock::now();
                mult_bloques_parallel(A, B, C, n, b);
                tb += duration<double>(high_resolution_clock::now() - t0).count();

                t0 = high_resolution_clock::now();
                strassen_parallel(A, B, n);
                tsh += duration<double>(high_resolution_clock::now() - t0).count();

                t0 = high_resolution_clock::now();
                strassen_puro_wrapper(A, B, n);
                tsp += duration<double>(high_resolution_clock::now() - t0).count();
            }
            tb  /= N_REPS;
            tsh /= N_REPS;
            tsp /= N_REPS;

            double sp_b  = T1_b  / tb;
            double sp_sh = T1_sh / tsh;
            double sp_sp = T1_sp / tsp;

            file << n << "," << t << ","
                 << tb  << "," << tsh  << "," << tsp  << ","
                 << sp_b  << "," << sp_b  / t << ","
                 << sp_sh << "," << sp_sh / t << ","
                 << sp_sp << "," << sp_sp / t << "\n";

            cout << "p=" << t
                 << " | Bloques: "   << tb  << "s (S=" << sp_b  << ")"
                 << " | Hibrido: "   << tsh << "s (S=" << sp_sh << ")"
                 << " | Puro: "      << tsp << "s (S=" << sp_sp << ")"
                 << endl;
        }
    }

    file.close();
    cout << "\nResultados guardados en resultados_parte2.csv" << endl;
    return 0;
}