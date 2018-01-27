#include <stdio.h>
#include <math.h>
#include <stack>
#include <stdlib.h>
#include <utility>
#include <iostream>

void matr_mul_l(double cos_, int i, double sin_, int j, double *matr, int size){
    double elem;
    for(size_t k = 0; k < size; ++k){
        elem = matr[i * size + k];
        matr[i * size + k] = matr[i * size + k] * cos_ - matr[j * size + k] * sin_;
        matr[j * size + k] = elem * sin_ + matr[j * size + k] * cos_;
    }
}

void matr_mul_r(double cos_, int i, double sin_, int j, double *matr, int size){
    double elem;
    for(size_t k = 0; k < size; ++k){
        elem = matr[k * size + i];
        matr[k * size + i] = matr[k * size + i] * cos_ + matr[k * size + j] * sin_;
        matr[k * size + j] = -elem * sin_ + matr[k * size + j] * cos_;
    }
}

void reverse_matr(double *matr, int size){
    double* res = (double*) malloc(size * size * sizeof(double));
    for (size_t i = 0; i < size; ++i){
        for (size_t j = 0; j < size; ++j){
            if (i == j) res[i * size + j] = 1;
            else res[i * size + j] = 0;
        }
    }
    
    for(size_t i = 0; i < size; ++i){
        for(size_t j = 0; j < size; ++j){
            if (i == j) res[i * size + j] = 1;
            else res[i * size + j] = 0;
        }
    }
    for(int i = size - 1; i >= 0; --i){
        for(int j = i - 1; j >= 0; --j){
            for(int k = size - 1; k >= i; --k){
                res[j * size + k] -= (matr[j * size + i] / matr[i * size + i]) *
                    res[i * size + k];
            }
        }
        for(int k = size - 1; k >= i; --k)
            res[i * size + k] *= 1 / matr[i * size + i];
    }
    for (size_t i = 0; i < size ; ++i){
        for (size_t j = 0; j < size; ++j){
            matr[i * size + j] = res[i * size + j];    
        }
    }
}

void mul_l(double* A, double* B, int size){
    double** matr = new double*[size];
    for (int i = 0; i < size; ++i)
        matr[i] = new double[size];
    for (int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j){
            double value = 0;
            for (int n = 0; n < size; ++n)
                value += A[i * size + n] * B[n * size + j];
            matr[i][j] = value;
        }
    }
    
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            A[i * size + j] = matr[i][j];
    
    for (int i = 0; i < size; ++i)
        delete [] matr[i];
    delete [] matr;
}

int lin_method(int i, int j, double *matr, int size){
    std::stack<std::pair<double, double> > angle;
    double* Q = new double[size * size];
    double* save = new double[size * size];
    for (int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j)
            save[i * size + j] = matr[i * size +j];
    }
    
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
            Q[i * size + j] = i == j ? 1 : 0;
    
     for (int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j)
            std::cout << Q[i * size + j] << " " ;
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    for (size_t i = 0; i < size - 1; ++i)
        for (size_t j = i + 1; j < size; ++j){
            if (fabs(matr[i * size + i]) < 1e-6 && fabs(matr[j * size + i]) < 1e-6){
                continue;
            }
            double cos_, sin_;
            cos_ = matr[i * size + i] /
                sqrt(pow(matr[i * size + i], 2) + pow(matr[j * size + i], 2));
            sin_ = -matr[j * size + i] /
                sqrt(pow(matr[i * size + i], 2) + pow(matr[j * size + i], 2));
            matr_mul_l(cos_, i, sin_, j, matr, size);
            angle.push(std::pair<double, double> (cos_, sin_));
            matr_mul_l(cos_, i, sin_, j, Q, size);
        }
    for (int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j)
            std::swap(Q[i * size + j], Q[j * size + i]);
    }
    
    
    mul_l(Q, save, size);
    for (int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j)
            std::cout << Q[i * size + j] - matr[i * size + j] << " ";
        std::cout << std::endl;
    }
    
    return 0;
}
