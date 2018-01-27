#include <utility>
#include <iostream>
#include <vector>
#include <fstream>

const int batch_count = 20;
const int max_n = 10000;
const int max_len = 50;
int dejikstra(std::vector<std::vector<std::pair<int, int> > >& g, int start, int end, bool update);

void convert(int** dist, std::vector<std::vector<std::pair<int, int> > >& g){
    int n = g.size();
    for (int i = 0; i <n; ++i)
        g[i].clear();
    
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j){
            if (dist[i][j] != -1){
                g[i].push_back(std::make_pair(j, dist[i][j]));
                g[j].push_back(std::make_pair(j, dist[j][i]));
            }
        }
}

void generate(int** dist, int size, double* time){
    int batch_size = size * (size - 1) / (2 * batch_count);
    
    for (size_t i = 0; i < (size_t)size; ++i){
        dist[i][i] = -1;
        for (size_t j = i + 1; j < (size_t)size; ++j){
            int len = 3 + rand() % (max_len - 3);
            dist[i][j] = dist[j][i] = len;
        }
        dist[i][size] = size - 1;
    }
    
    std::vector<std::vector<std::pair<int, int> > > g(size);
    convert(dist, g);
    
    double t = clock();
    dejikstra(g, 0, size / 2, 1);
    time[0] += (double)(clock() - t) / 1e6;
    
    for (size_t it = 0; it < (size_t)batch_count - 1; ++it){
        for (size_t i = 0; i < (size_t)batch_size; ++i){
            
            int from = rand() % size;
            while (dist[from][size] == 2)
                from = rand() % size;
    
            int to = 0; 
            while (dist[from][to] == -1 || (to + 1) % size == from || (from + 1) % size == to)
                ++to;
            
            if (to >= size){
                std::cout << "Out of range" << std::endl;
                return;
            }
            
            dist[from][to] = dist[to][from] = -1;
            --dist[from][size];
            --dist[to][size];
        }
        convert(dist, g);
        
        t = clock();
        dejikstra(g, 0, size / 2, 1);
        time[it + 1] += (double)(clock() - t) / 1e6;
    }
}

int main(int argc, char** argv){
    
    srand(10);
    std::ofstream fout;
    int count = std::stoi(argv[1]);
    int** dist = new int*[max_n];
    double* time = new double[batch_count];
        
    fout.open("res.txt");
    for (size_t i = 0; i < max_n; ++i)
        dist[i] = new int[max_n + 1];
    
    for (size_t i = 0; i < (size_t)count; ++i){
        int size = 600 + 100 * i;
        
        for (size_t j = 0; j < (size_t)batch_count; ++j)
            time[j] = 0;
        
        for (size_t j = 0; j < 10; ++j){
            generate(dist, size, time);
        }
        
        fout << size << "\t";
        std::cout << "n:" << size << "\t";
        for (size_t j = 0; j < (size_t)batch_count; ++j){
            fout << time[j] << "\t";
            std::cout << time[j] << "\t";
        }
        fout << std::endl;
        std::cout << std::endl;
    } 
    
    for (int i = 0; i < max_n; ++i)
        delete [] dist[i];
    delete [] dist;
    delete [] time;
    
    fout.close();
    return 0;
}