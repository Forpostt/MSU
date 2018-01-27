#include <stdio.h>
#include <math.h>
#include <stack>
#include <stdlib.h>
#include <utility>
#include <pthread.h>

typedef struct matrix_multiplication {
    int num; 
    int t_quantity; 
    int n; 
    double *matrix;
    double *reverse;
    double *result;
} m_m;

void matr_mul_l(double cos_, int i, double sin_, int j, double *matr, int size){
    double elem;
    for(size_t k = i; k < size; ++k){
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

int lin_method(double *matr, int size){
    std::stack<std::pair<double, double> > angle;
    for (size_t i = 0; i < size - 1; ++i){
        for (size_t j = i + 1; j < size; ++j){
            if (fabs(matr[i * size + i]) < 1e-6 && fabs(matr[j * size + i]) < 1e-6){
                angle.push(std::pair<double, double> (1., 0.));
                continue;
            }
            double cos_, sin_;
            cos_ = matr[i * size + i] /
                sqrt(pow(matr[i * size + i], 2) + pow(matr[j * size + i], 2));
            sin_ = -matr[j * size + i] /
                sqrt(pow(matr[i * size + i], 2) + pow(matr[j * size + i], 2));
            matr_mul_l(cos_, i, sin_, j, matr, size);
            angle.push(std::pair<double, double> (cos_, sin_));
        }
    }
    
    double sum = 0;
    for (size_t i = 0; i < size; ++i){
        sum += matr[(size - 1) * size +i];
    }
    if (fabs(sum) < 1e-6) return -1; 
    
    reverse_matr(matr, size);
    for (int i = size - 2; i >= 0; --i){
        for (int j = size - 1; j > i; --j){
            matr_mul_r(angle.top().first, i, angle.top().second, j, matr, size);
            angle.pop();
        }
    }
    return 0;
}

void generate_matr(double* matr, double* save, int size){
    for (size_t i = 0; i < size; ++i){
        for (size_t j = 0; j < size; ++j){
            if (i == j && j == 0) {
                matr[i * size + j] = -1.;
                save[i * size + j] = -1.;
            }
            else if (i == j  && j == size - 1){ 
                matr[i * size + j] = -(size - 1.) / size;
                save[i * size + j] = -(size - 1.) / size;
            }
            else if (i == j){ 
                matr[i * size + j] = -2.;
                save[i * size + j] = -2.;
            }
            else if (i - j == 1 || j - i == 1){
                matr[i * size + j] = 1;
                save[i * size + j] = 1;
            }
            else{
                 matr[i * size + j] = 0;
                 save[i * size + j] = 0;
            }
        }
    }
}

void *norma(void *args) {
    m_m *mult = (m_m *) args;
    int n = mult->n;
    int num = mult->num;
    int k = mult->t_quantity;
    printf("\n%d\n", num);
    if (num != k) {
        for (int i = (num - 1) * (n / k); i < num * (n / k); i++)
            for (int j = 0; j < n; j++)
                for (int l = 0; l < n; l++)
                    mult->result[i * n + j] += mult->matrix[i * n + l] * mult->reverse[l * n + j];
        return 0;
    }
    for (int i = (k - 1) * (n / k); i < n; i++)
        for (int j = 0; j < n; j++)
            for (int l = 0; l < n; l++)
                mult->result[i * n + j] += mult->matrix[i * n + l] * mult->reverse[l * n + j];
    return 0;
}


double find_norma(int n, double *for_norma){
    for (int i = 0; i < n; i++)
        for_norma[i * n + i] -= 1.0;

    double max = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            max += for_norma[i * n + j] * for_norma[i * n + j];

    return sqrt(max);
}
