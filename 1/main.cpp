#include <stdio.h>
#include <stdlib.h>
#include <string>

int lin_method(int i, int j, double *matr, int size);

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

int main(int argc, char* argv[]){
    FILE *fin, *fout;
    fin = fopen("input.txt", "r");
    fout = fopen("output.txt", "w");
    int size;
    if (argc == 1) fscanf(fin, "%d", &size);
    else size = std::stoi(std::string(argv[1]));
    double *matr = (double *)malloc(size * size * sizeof(double));
    double *save = (double *)malloc(size * size *sizeof(double));
    if (argc != 1) generate_matr(matr, save, size);
    else{
        for (size_t i = 0; i < size * size; ++i){
            double x;
            if (fscanf(fin, "%lf", &x) != 1){
                fprintf(stderr, "Wrong %d elemnt\n", (int)i);
                return -1;
            }
            save[i] = matr[i] = x;
        }
    }
    if (lin_method(0, 1, matr, size) != 0){
        fprintf(fout, "Singular +matrix!\n");
        return 0;
    }
    
    /*fprintf(fout, "Result matrix:\n");
    for (size_t i = 0; i < size; ++i){
        for (size_t j = 0; j < size; ++j){
            fprintf(fout, "%lf ", matr[i * size + j]);
        }
        fprintf(fout, "\n");
    }
    
    fprintf(fout, "Norms:\n");
    for(size_t i = 0; i < size; ++i){
        for(size_t j = 0; j < size; ++j){
            double elem = 0;
            for(size_t k = 0; k < size; ++k){
                elem += save[i * size + k] * matr[k * size + j];
            }
            if (i == j) elem -= 1;
            fprintf(fout, "%lf ", elem);
        }
        fprintf(fout, "\n");
    }
    */
    return 0;
}
