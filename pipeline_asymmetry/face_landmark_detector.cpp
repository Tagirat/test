#include "face_landmark_detector.h"
#include <dlib/image_io.h> 

FaceLandmarkDetector::FaceLandmarkDetector(const std::string &model_path) {
  detector = dlib::get_frontal_face_detector();
  dlib::deserialize(model_path) >> predictor;
}

std::vector<dlib::full_object_detection>
FaceLandmarkDetector::detect(const dlib::array2d<dlib::rgb_pixel> &img) {
  std::vector<dlib::rectangle> faces = detector(img);
  std::vector<dlib::full_object_detection> shapes;

  for (const auto &face : faces) {
    shapes.push_back(predictor(img, face));
  }
  return shapes;
}
