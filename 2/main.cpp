#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int reflect(double** matr, int size);
int eigenvalue(double** matr, int size, double s);

int main(){
    FILE *fin, *fout;
    fin = fopen("input.txt", "r");
    fout = fopen("output.txt", "w");
    int size;
    fscanf(fin, "%d", &size);
    
    double** matr = new double*[size];
    double** save = new double*[size];
    double norm = 0.;
    for (int i = 0; i < size; ++i){
        matr[i] = new double[size];
        save[i] = new double[size];
    }
    
    for (int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j){
            double x;
            if (fscanf(fin, "%lf", &x) != 1){
                fprintf(stderr, "Wrong %d elemnt\n", (int)i);
                return -1;
            }
        matr[i][j] = save[i][j] = x;
        if (i == j)
            norm += x;
        }
    }
    double sphere_norm = 0.;
    
    for (int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j)
            sphere_norm += (matr[i][j]) * (matr[i][j]);
    }
    fprintf(fout, "Sphere norm: %lf\n", sphere_norm);
    
    reflect(matr, size);
    sphere_norm = 0.;    
    for (int i = 0; i < size; ++i){
        for (int j = 0; j < size; ++j)
            sphere_norm += (matr[i][j]) * (matr[i][j]);
    }
    
    fprintf(fout, "New sphere norm: %lf\n", sphere_norm);
    
    if (eigenvalue(matr, size, 0.05) != 0){
        std::cerr << "Singular matrix" << std::endl;
        return 0;
    }
    
    fprintf(fout, "Eigen values: \n");
    for (size_t i = 0; i < size; ++i){
        fprintf(fout, "%lf ", matr[i][i]);
    }
    
    double nor = 0.;
    for (int i = 0; i < size; ++i)
        norm -= matr[i][i]; 
    fprintf(fout, "\nNorm: \n%lf", nor);
    return 0;
}