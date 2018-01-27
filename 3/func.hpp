#ifndef FUNCTIONS_H
#define FUNCTIONS_H

typedef struct matrix_multiplication {
    int num; 
    int t_quantity; 
    int n; 
    double *matrix;
    double *reverse;
    double *result;
} m_m;

int lin_method(double *matr, int size);

void generate_matr(double* matr, double* save, int size);

void *norma(void *args);

double find_norma(int n, double *for_norma);

#endif