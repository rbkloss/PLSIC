#pragma once
#include <string>
#include <opencv2/core.hpp>

#include <ml/clustering.hpp>
#include <ml/classifier_clustering.hpp>
#include <ml/pls_image_clustering.hpp>
#include <memory>



class VideoSummarization{
  bool viewBegin_ = false;
  bool viewPartial_ = false;
  bool viewFinal_ = false;
  bool responses_ = false;

  char* featPath = "d:/Docs/Development/DataSets/OpenVideos/features/colorCM/08x08/";
  char* videoPath = "d:/Docs/Development/DataSets/OpenVideos/database/";
  char* summaryPath = "summary/";
  /*char* featPath = "u:/Videos/features/colorCM/08x08/";
  char* videoPath = "u:/Videos/database/";
  char* summaryPath = "u:/Videos/summary/";*/

  int frameSampling_ = 5;
  std::vector<int> frameIds_;
  std::vector<cv::Mat> videoFrames_;

  std::shared_ptr<ssig::PLSImageClustering> configurePlsic(const std::string& plsicConfig,
                                                          const std::string& initConfig,
                                                          const int videoId,
                                                          cv::Mat_<float>& features);
  std::vector<ssig::Cluster> pickKeyframes(const cv::Mat& feats,
                                          const std::vector<ssig::Cluster>& clusters,
                                          const std::vector<std::vector<float>>& clustersResponses);

  void readFeatures(const std::string& filename,
                    std::vector<int>& frameIds,
                    cv::Mat_<float>& features);

  std::vector<ssig::Cluster> cluster(cv::Mat_<float>& inp,
                                    ssig::Clustering& clusteringMethod);
  std::vector<ssig::Cluster> cluster(const int videoid,
                                    cv::Mat_<float>& inp,
                                    ssig::ClassifierClustering& clusteringMethod);

  void visualizeClusters(const std::vector<ssig::Cluster>& clustering,
                         std::vector<cv::Mat>& visualization);

  void loadVideo(const std::string& videoname,
                 std::vector<cv::Mat>& videoFrames);

public:
  VideoSummarization(const int sampling, const bool selectKeyframesResponses);

  void executePlsic(const std::string& plsicConfig,
                    const std::string& initConfig);
  void executeKmeans(const std::string& kmeansConfig,
                     const std::string& initConfig);
};
