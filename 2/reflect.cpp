#include <iostream>
#include <math.h>

void shift_reflect_matr(double** matr, double** U, int iter, int size){
    double norm, new_norm;
    double* vec = new double[size];
    for (int i = iter + 1; i < size; ++i)
        norm += matr[i][iter] * matr[i][iter];
    
    new_norm = sqrt(2 * norm - 2 * sqrt(norm) * matr[iter + 1][iter]);
    norm = sqrt(norm); 
    
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            U[i][j] = i == j ? 1 : 0;
    
    if (new_norm < 1e-6)
        return;
    
    for (int i = iter + 1; i < size; ++i)
        vec[i] = i == iter + 1 ? (matr[i][iter] - norm) / new_norm : matr[i][iter] / new_norm;
    
    for (int i = iter + 1; i < size; ++i)
        for (int j = i; j < size; ++j)
            U[i][j] = U[j][i] = i == j ? (1 - 2 * vec[i] * vec[j]) : (- 2 * vec[i] * vec[j]);  
    delete []vec;
}

void mul_r(double** A, double** B, int size){
    double** matr = new double*[size];
    for (int i = 0; i < size; ++i)
        matr[i] = new double[size];
    for (int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j){
            double value = 0;
            for (int n = 0; n < size; ++n)
                value += A[i][n] * B[n][j];
            matr[i][j] = value;
        }
    }
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            B[i][j] = matr[i][j];
    for (int i = 0; i < size; ++i)
        delete [] matr[i];
    delete [] matr;
}

void mul_l(double** A, double** B, int size){
    double** matr = new double*[size];
    for (int i = 0; i < size; ++i)
        matr[i] = new double[size];
    for (int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j){
            double value = 0;
            for (int n = 0; n < size; ++n)
                value += A[i][n] * B[n][j];
            matr[i][j] = value;
        }
    }
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            A[i][j] = matr[i][j];
    
    for (int i = 0; i < size; ++i)
        delete [] matr[i];
    delete [] matr;
}
/*
void reflect_mul(double** matr, double** U, int iter, int size){
    double* vec = new double[size];
    for (int j = iter; j < size; ++j){
        for (int i = iter + 1; i < size; ++i){
            double value = 0;
            for (int n = iter; n < size; ++n)
                value += U[i][n] * matr[n][j];
            vec[i] = value;
        }
        for (int i = iter + 1; i < size; ++i)
            matr[i][j] = vec[i];
    }
    delete []vec;
}
*/
void reflect(double** matr, int size){
    double**  U = new double*[size];
    for (int i = 0; i < size; ++i)
        U[i] = new double[size];
    
    for (int iter = 0; iter < size - 2; ++iter){
        shift_reflect_matr(matr, U, iter, size);
        mul_r(U, matr, size);    
    }
    
    for (int i = 0; i < size; ++i)
        delete [] U[i];
    delete [] U;
}