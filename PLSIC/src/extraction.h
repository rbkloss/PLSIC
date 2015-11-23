#include <string>
#include <opencv2/core.hpp>


class Extraction{
public:
  static void extractFeatures(const std::string& structureFile,
                              const std::string& outDir,
                              const cv::Size& windowSize,
                              const std::pair<float, float>& stride);

  static void writeDescriptor(
    const std::vector<cv::Mat_<float>>& descriptors,
    const std::vector<cv::Rect>& rects,
    const std::vector<float>& scales,
    const std::string& outDir, std::string& outputname);

  static void readFeatures(std::string& featurefilename,
                           cv::Mat_<float>& features,
                           std::string& imagename,
                           std::vector<cv::Rect>& patches,
                           std::vector<float> &scales);
};
