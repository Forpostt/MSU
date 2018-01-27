#include <iostream>
#include <math.h>

void mul_l(double** A, double** B, int size);
void mul_r(double** A, double** B, int size);

void reflect_matr(double** matr, double** U, int iter, int size){
    double norm, new_norm;
    double* vec = new double[size];
    for (int i = iter; i < size; ++i)
        norm += matr[i][iter] * matr[i][iter];
    
    new_norm = sqrt(2 * norm - 2 * sqrt(norm) * matr[iter][iter]);
    norm = sqrt(norm); 
    
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            U[i][j] = i == j ? 1 : 0;

    if (new_norm < 1e-6)
        return;
        
    for (int i = iter; i < size; ++i)
        vec[i] = i == iter ? (matr[i][iter] - norm) / new_norm : matr[i][iter] / new_norm;
    
    for (int i = iter; i < size; ++i)
        for (int j = i; j < size; ++j)
            U[i][j] = U[j][i] = i == j ? (1 - 2 * vec[i] * vec[j]) : (- 2 * vec[i] * vec[j]);  
    
    delete []vec;
}

void qr(double** R, double** Q, int size){
    double** U = new double*[size];
    for (int i = 0; i < size; ++i)
        U[i] = new double[size];
    
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            Q[i][j] = i == j ? 1 : 0;
    
    for (int iter = 0; iter < size - 1; ++iter){
        reflect_matr(R, U, iter, size);
        mul_r(U, R, size);
        mul_l(Q, U, size);
    }
}

int eigenvalue(double** matr, int size, double s){
    double** Q = new double*[size];
    double* old_eige = new double[size];
    for (int i = 0; i < size; ++i){
        Q[i] = new double[size];
        old_eige[i] = matr[i][i];
    }
    double error = 1.;
    int iter = 1;
    while (error > 1e-6){
        for (int i = 0; i < size; ++i){
            for (int j = 0; j < size; ++j)
                matr[i][j] = i == j ? matr[i][j] - s : matr[i][j];
        }
        
        qr(matr, Q, size);
        
        error = 0.;
        for (int i = 0; i < size; ++i)
            error  += matr[size - 1][i];
        if (-1e-6 < error && error < 1e-6)
            return -1;
        
        mul_l(matr, Q, size);
        for (int i = 0; i < size; ++i){
            for (int j = 0; j < size; ++j)
                matr[i][j] = i == j ? matr[i][j] + s : matr[i][j];
        }
        error = 0;
        for (int i = 0; i < size; ++i){
            error += (matr[i][i] - old_eige[i]) * (matr[i][i] - old_eige[i]);
            old_eige[i] = matr[i][i];
        }
        error = sqrt(error);
        
    }
    return 0;
}