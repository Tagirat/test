#include "face_asymmetry_analyzer.h"
#include <cmath>
#include <codecvt>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

// ANSI коды цветов
#define COLOR_RESET "\033[0m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"

std::string get_ratio_color(double ratio) {
  if (std::abs(ratio - 1.0) < 0.05) {
    return COLOR_GREEN; // Почти идеально (зеленый)
  } else if (std::abs(ratio - 1.0) < 0.1) {
    return COLOR_YELLOW; // Умеренное отклонение (желтый)
  } else {
    return COLOR_RED; // Сильное отклонение (красный)
  }
}

std::vector<FaceAsymmetryAnalyzer::AsymmetryResult>
FaceAsymmetryAnalyzer::analyze(const dlib::full_object_detection &shape) {
  std::vector<AsymmetryResult> results;

  // Центральная точка (57)
  const dlib::point center = shape.part(57);

  // Пары измерений: {название, левая точка, правая точка, центр (если
  // отличается)}
  const std::vector<std::tuple<std::string, int, int, int>> measurements = {
      {"От наивысшей точки брови до центра нижней губы", 20, 23, 57},
      {"От середины брови, до центра нижней губы", 18, 25, 57},
      {"От внешнего края глаза, до центра нижней губы", 36, 45, 57},
      {"От нижней части носа, до края губ", 33, 33,
       0}, // 0 означает что будем использовать 48 и 54 как правые точки
      {"От внешнего края глаза, до внутреннего края глаза", 36, 45, 39}, // 39 и 42 - внутренние уголки глаз
      {"От ноздри к нижнему, до центра нижней губы", 31, 35, 57},
      {"От края губ, до центра нижней губы", 48, 54, 57},
      {"От края глаз, до крыльев носа", 36, 45, 31}};

  for (const auto &[name, left, right, center_point] : measurements) {
    AsymmetryResult res;
    res.measurement_name = name;

    // Для особого случая с губами (33 точка носа)
    if (left == 33 && right == 33) {
      res.left_distance = dlib::length(shape.part(left) - shape.part(48));
      res.right_distance = dlib::length(shape.part(right) - shape.part(54));
    }
    // Для глаз (горизонталь)
    else if (center_point == 39) {
      res.left_distance = dlib::length(shape.part(left) - shape.part(39));
      res.right_distance = dlib::length(shape.part(right) - shape.part(42));
    }
    // Все остальные измерения относительно центра
    else {
      res.left_distance = dlib::length(shape.part(left) - center);
      res.right_distance = dlib::length(shape.part(right) - center);
    }

    res.ratio = res.left_distance / res.right_distance;
    results.push_back(res);
  }

  return results;
}

// Функция для вычисления реальной длины строки с учетом UTF-8
size_t utf8_strlen(const std::string &str) {
  size_t length = 0;
  for (char c : str) {
    if ((c & 0xC0) != 0x80)
      length++;
  }
  return length;
}

// Функция для создания строки с нужной шириной (дополняет пробелами справа)
std::string pad_right(const std::string &str, size_t width) {
  size_t str_len = utf8_strlen(str);
  if (str_len >= width)
    return str;
  return str + std::string(width - str_len, ' ');
}

void FaceAsymmetryAnalyzer::printResults(
    const std::vector<AsymmetryResult> &results) {
  // Ширина колонок
  const size_t name_width = 50; // Ширина колонки с названием
  const size_t value_width = 16; // Ширина числовых колонок
  const size_t ratio_width = 12; // Уменьшенная ширина для отношения

  // Символы для рамки
  const std::string top_left = "┌";
  const std::string top_center = "┬";
  const std::string top_right = "┐";
  const std::string middle_left = "├";
  const std::string middle_center = "┼";
  const std::string middle_right = "┤";
  const std::string bottom_left = "└";
  const std::string bottom_center = "┴";
  const std::string bottom_right = "┘";
  const std::string vertical_line = "│";
  const std::string horizontal_line = "─";

  std::cout << "\n\nАнализ асимметрии лица\n";

  // Верхняя граница таблицы
  std::cout << top_left << std::string(name_width, '-') << top_center
            << std::string(value_width, '-') << top_center
            << std::string(value_width, '-') << top_center
            << std::string(ratio_width, '-') << top_right << "\n";

  // Заголовки
  std::cout << vertical_line << pad_right(" Параметр", name_width)
            << vertical_line << pad_right(" Левая сторона", value_width)
            << vertical_line << pad_right(" Правая сторона", value_width)
            << vertical_line << pad_right(" Отношение", ratio_width)
            << vertical_line << "\n";

  // Разделительная линия
  std::cout << middle_left << std::string(name_width, '-') << middle_center
            << std::string(value_width, '-') << middle_center
            << std::string(value_width, '-') << middle_center
            << std::string(ratio_width, '-') << middle_right << "\n";

  // Вывод данных
  for (const auto &res : results) {
    std::string color = get_ratio_color(res.ratio);

    std::string left_val = std::to_string(res.left_distance);
    left_val = left_val.substr(0, left_val.find(".") + 3);

    std::string right_val = std::to_string(res.right_distance);
    right_val = right_val.substr(0, right_val.find(".") + 3);

    std::string ratio_val = std::to_string(res.ratio);
    ratio_val = ratio_val.substr(0, ratio_val.find(".") + 3);
    std::cout << vertical_line << " "
              << pad_right(res.measurement_name, name_width - 1)
              << vertical_line << " " << pad_right(left_val, value_width - 1)
              << vertical_line << " " << pad_right(right_val, value_width - 1)
              << vertical_line << " "
              << get_ratio_color(res.ratio) // Устанавливаем цвет
              << pad_right(ratio_val, ratio_width - 1) // Выводим значение
              << COLOR_RESET << vertical_line << "\n";
    // std::cout << vertical_line << " " << pad_right(res.measurement_name,
    // name_width - 1)
    //           << vertical_line << " " << pad_right(left_val, value_width - 1)
    //           << vertical_line << " " << pad_right(right_val, value_width -
    //           1)
    //           << vertical_line << " " << pad_right(ratio_val, ratio_width -
    //           1) << vertical_line << "\n";
  }

  // Нижняя граница таблицы
  std::cout << bottom_left << std::string(name_width, '-') << bottom_center
            << std::string(value_width, '-') << bottom_center
            << std::string(value_width, '-') << bottom_center
            << std::string(ratio_width, '-') << bottom_right << "\n";
}

void FaceAsymmetryAnalyzer::saveToCSV(
    const std::vector<AsymmetryResult> &results,
    const std::string &base_output_dir, const std::string &image_filename) {
  // Создаем папку для результатов, если её нет
  if (!fs::exists(base_output_dir)) {
    fs::create_directory(base_output_dir);
  }

  // Формируем имя CSV-файла на основе имени изображения
  std::string csv_filename = base_output_dir + "/" +
                             fs::path(image_filename).stem().string() +
                             "_analysis.csv";

  std::ofstream outfile(csv_filename);

  // Заголовок CSV
  outfile << "Measurement,Left Distance,Right Distance,Ratio (L/R)\n";

  // Записываем данные
  for (const auto &res : results) {
    outfile << "\"" << res.measurement_name << "\"," << res.left_distance << ","
            << res.right_distance << "," << res.ratio << "\n";
  }

  std::cout << "Результаты сохранены в: " << csv_filename << std::endl;
}
