#ifndef FACE_LANDMARK_DETECTOR_H
#define FACE_LANDMARK_DETECTOR_H

#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <string>
#include <vector>

class FaceLandmarkDetector {
private:
  dlib::frontal_face_detector detector;
  dlib::shape_predictor predictor;

public:
  FaceLandmarkDetector(const std::string &model_path);
  std::vector<dlib::full_object_detection>
  detect(const dlib::array2d<dlib::rgb_pixel> &img);
};

#endif // FACE_LANDMARK_DETECTOR_H
