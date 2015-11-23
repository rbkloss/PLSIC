#ifndef _REID_H_
#define _REID_H_
#include <string>
#include <opencv2/core.hpp>


class Reid{
  static void extract(const std::string& filename, const int cols, cv::Mat_<float>& features);
  void clusterPlsic(const cv::Mat_<float>& inp)const;
  void clusterKmeans(const cv::Mat_<float>& inp)const;
  void visualize(const std::vector<int>& cluster, const std::string& path, cv::Mat& img) const;
public:
  void execute()const;
};

#endif
