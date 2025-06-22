#include "face_landmark_detector.h"
#include "image_processor.h"
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

int main() {
  // Инициализация детектора
  FaceLandmarkDetector detector("shape_predictor_68_face_landmarks.dat");

  // Инициализация процессора изображений
  ImageProcessor processor(detector, "output");

  // Обработка всех изображений в папке dataset
  for (const auto &entry : fs::directory_iterator("dataset")) {
    if (entry.path().extension() == ".jpg" ||
        entry.path().extension() == ".png") {
      processor.process_image(entry.path().string());
    }
  }

  std::cout << "Processing complete. Results saved to 'output' directory."
            << std::endl;
  return 0;
}
