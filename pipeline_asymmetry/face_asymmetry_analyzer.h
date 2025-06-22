#ifndef FACE_ASYMMETRY_ANALYZER_H
#define FACE_ASYMMETRY_ANALYZER_H

#include <dlib/image_processing.h>
#include <fstream>
#include <string>
#include <vector>

class FaceAsymmetryAnalyzer {
public:
  struct AsymmetryResult {
    std::string measurement_name;
    double left_distance;
    double right_distance;
    double ratio; // left/right
  };

  static std::vector<AsymmetryResult>
  analyze(const dlib::full_object_detection &shape);
  static void printResults(const std::vector<AsymmetryResult> &results);
  static void saveToCSV(const std::vector<AsymmetryResult> &results,
                        const std::string &base_output_dir,
                        const std::string &image_filename);
};

#endif // FACE_ASYMMETRY_ANALYZER_H
