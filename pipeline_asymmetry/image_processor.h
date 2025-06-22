#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include "face_landmark_detector.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>

namespace fs = std::filesystem;

class ImageProcessor {
private:
  FaceLandmarkDetector &detector;
  std::string output_dir;

  void draw_landmarks(cv::Mat &image, const dlib::full_object_detection &shape);

public:
  ImageProcessor(FaceLandmarkDetector &det, const std::string &out_dir);
  void process_image(const std::string &input_path);
};

#endif // IMAGE_PROCESSOR_H
