#include <iostream>

void find_path(int n, int m, int* path, int* path_size) {
    int current = 1;
    *path_size = 0;
    *(path + *path_size) = current;
    *path_size = *path_size + 1;
    while (1) {
        current = (current + m - 1) % n;
        if (current == 0) current = n;
        if (current == 1) break;
        *(path + *path_size) = current;
        *path_size = *path_size + 1;
    }
}

int main() {
    int n, m;
    std::cout << "Enter n: ";
    std::cin >> n;
    std::cout << "Enter m: ";
    std::cin >> m;
    int* path = (int*)malloc(n * sizeof(int));
    int path_size = 0;
    find_path(n, m, path, &path_size);
    std::cout << "Path: ";
    for (int i = 0; i < path_size; i++) {
        std::cout << *(path + i);
    }
    std::cout << "\n";
    free(path);

    return 0;
}
