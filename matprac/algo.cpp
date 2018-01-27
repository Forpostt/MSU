#include <iostream>
#include <vector>
#include <utility>

const int INF = 100000000;

int dejikstra(std::vector<std::vector<std::pair<int, int> > >& g, int start, int end, bool update){
    int size = g.size();
    std::vector<int> dist(size, INF);
    dist[start] = 0;
    std::vector<bool> u(size, 0);
    for (size_t i = 0; i < (size_t)size ; ++i){
        int v = -1;
        for (size_t j = 0; j < (size_t)size; ++j)
            if (!u[j] && (v == -1 || dist[j] < dist[v]))
                v = j;
        if (dist[v] == INF)
            break;
        
        if (update && v == end)
            return dist[end];
        
        u[v] = 1;
        for (size_t j = 0; j < g[v].size(); ++j){
            int to = g[v][j].first;
            int len = g[v][j].second;
            if (dist[to] > dist[v] + len)
                dist[to] = dist[v] + len;
        
        }
    }
    return dist[end];
}