#include "image_processor.h"
#include "face_asymmetry_analyzer.h"
#include <dlib/image_io.h> 
#include <dlib/opencv.h>   
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

ImageProcessor::ImageProcessor(FaceLandmarkDetector &det,
                               const std::string &out_dir)
    : detector(det), output_dir(out_dir) {
  fs::create_directory(output_dir);
}

void ImageProcessor::draw_landmarks(cv::Mat &image,
                                    const dlib::full_object_detection &shape) {
  // Цвета для разных групп точек
  const std::map<int, cv::Scalar> landmark_colors = {
      {0, cv::Scalar(0, 255, 0)},    // Челюсть
      {17, cv::Scalar(0, 0, 255)},   // Брови
      {27, cv::Scalar(255, 0, 0)},   // Нос
      {36, cv::Scalar(0, 255, 255)}, // Глаза
      {48, cv::Scalar(255, 0, 255)}  // Губы
  };

  // Рисуем точки и номера
  for (unsigned int i = 0; i < shape.num_parts(); ++i) {
    cv::Point pt(shape.part(i).x(), shape.part(i).y());

    // Выбираем цвет в зависимости от группы точек
    cv::Scalar color = cv::Scalar(200, 200, 200); // По умолчанию серый
    for (const auto &[start_idx, col] : landmark_colors) {
      if (i >= start_idx)
        color = col;
    }

    // Рисуем точку
    cv::circle(image, pt, 3, color, -1);

    // Рисуем номер точки
    cv::putText(image, std::to_string(i), cv::Point(pt.x + 7, pt.y - 7),
                cv::FONT_HERSHEY_DUPLEX, 0.5,
                cv::Scalar(255, 255, 255), // Белый текст
                1, cv::LINE_AA);
  }

  // Рисуем линии между точками челюсти (зеленые)
  for (unsigned int i = 0; i < 16; ++i) {
    cv::line(image, cv::Point(shape.part(i).x(), shape.part(i).y()),
             cv::Point(shape.part(i + 1).x(), shape.part(i + 1).y()),
             cv::Scalar(0, 255, 0), 1);
  }

  // Первый набор соединений (синие линии)
  const std::vector<std::pair<int, int>> blue_connections = {
      {36, 39}, {42, 45}, {36, 31}, {45, 35}, {36, 48},
      {45, 54}, {31, 57}, {35, 57}, {48, 57}, {57, 54}};

  for (const auto &[p1, p2] : blue_connections) {
    cv::line(image, cv::Point(shape.part(p1).x(), shape.part(p1).y()),
             cv::Point(shape.part(p2).x(), shape.part(p2).y()),
             cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
  }

  // Второй набор соединений (красные линии)
  const std::vector<std::pair<int, int>> red_connections = {
      {20, 57}, {23, 57}, {18, 57}, {25, 57},
      {36, 57}, {45, 57}, {33, 48}, {33, 54}};

  for (const auto &[p1, p2] : red_connections) {
    cv::line(image, cv::Point(shape.part(p1).x(), shape.part(p1).y()),
             cv::Point(shape.part(p2).x(), shape.part(p2).y()),
             cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
  }

  // Выделяем центральную точку 57 (желтый)
  cv::Point center_pt(shape.part(57).x(), shape.part(57).y());
  cv::circle(image, center_pt, 5, cv::Scalar(0, 255, 255), -1);
  cv::putText(image, "57", cv::Point(center_pt.x + 10, center_pt.y - 10),
              cv::FONT_HERSHEY_DUPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
}

void ImageProcessor::process_image(const std::string &input_path) {
  try {
    dlib::array2d<dlib::rgb_pixel> img;
    dlib::load_image(img, input_path);

    FaceLandmarkDetector detector("shape_predictor_68_face_landmarks.dat");
    auto shapes = this->detector.detect(img);

    if (shapes.empty()) {
      std::cout << "No faces found in " << input_path << std::endl;
      return;
    }

    cv::Mat cv_img = dlib::toMat(img);

    for (const auto &shape : shapes) { // Добавляем объявление shape
      draw_landmarks(cv_img, shape);

      // Анализ асимметрии
      auto results = FaceAsymmetryAnalyzer::analyze(shape);
      std::cout << "\nАнализ для файла: " << input_path << std::endl;
      FaceAsymmetryAnalyzer::printResults(results);
      std::string image_name = fs::path(input_path).filename().string();
      FaceAsymmetryAnalyzer::saveToCSV(results, "output_csv", image_name);
    }

    std::string output_path =
        "output/annotated_" +
        std::filesystem::path(input_path).filename().string();
    cv::imwrite(output_path, cv_img);
    std::cout << "Processed: " << input_path << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error processing " << input_path << ": " << e.what()
              << std::endl;
  }
}
