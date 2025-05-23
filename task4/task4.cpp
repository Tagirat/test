#include <cstdlib>
#include <fstream>
#include <iostream>

void bubble_sort(int *arr, int size) {
  for (int i = 0; i < size - 1; ++i) {
    for (int j = 0; j < size - i - 1; ++j) {
      if (arr[j] > arr[j + 1]) {
        int temp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = temp;
      }
    }
  }
}

int *read_from_file(const char *filename, int &out) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error of oppening file" << filename << std::endl;
    exit(1);
  }

  int count = 0;
  int temp;
  while (file >> temp) {
    count++;
  }
  file.clear();
  file.seekg(0);

  int *numbers = new int[count];

  int index = 0;
  while (file >> numbers[index]) {
    index++;
  }

  out = count;
  file.close();
  return numbers;
}

int main() {
  const char *filename = "file.txt";

  int size;
  int *nums = read_from_file(filename, size);

  if (size == 0) {
    std::cout << 0 << std::endl;
    delete[] nums;
    return 0;
  }

  bubble_sort(nums, size);

  int median = nums[size >> 1];

  int moves = 0;
  for (int i = 0; i < size; i++) {
    moves += abs(nums[i] - median);
  }

  std::cout << moves << std::endl;

  delete[] nums;

  return 0;
}
