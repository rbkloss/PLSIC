#ifndef _CLUSTERING_H_
#define _CLUSTERING_H_

#include <ml/clustering.hpp>
#include <ml/kmeans.hpp>
#include <ml/classifier_clustering.hpp>
#include <ml/pls_image_clustering.hpp>
#include <ml/singh.hpp>


class Clustering {
public:
  virtual ~Clustering() = default;

  void execute(ssig::PLSImageClustering& plsic,
               const std::string& initializationMode,
               const std::string& path,
               const std::string& modelname);
  void execute(ssig::Singh& singh,
               const std::string& path,
               const std::string& pathNegatives,
               const std::string& modelname);
  void execute(ssig::Kmeans& kmeans,
               const std::string& path,
               const std::string& modelname) const;


  void setup(const std::string& pathToFeatures,
             cv::Mat_<float>& inp,
             std::vector<std::string>& sampleImagenames,
             std::vector<cv::Rect>& samplePatches,
             std::vector<float>& sampleScales) const;

  void setup(const std::string& pathToFeatures,
             cv::Mat_<float>& inp) const;

  static std::vector<std::vector<int>> split(cv::Mat_<float>& inp);


  static void cluster(cv::Mat_<float>& input, ssig::Clustering& clustering);
  static void cluster(cv::Mat_<float>& input,
               const std::vector<std::string>& mapSampleImagename,
               const std::vector<cv::Rect>& mapSamplePatch,
               std::vector<float>& sampleScales,
               ssig::ClassifierClustering& clustering);

  static void visualizeCluster(const std::vector<int>& cluster,
                               const std::vector<std::string>& mapSampleImagename,
                               const std::vector<cv::Rect>& mapSamplePatch,
                               std::vector<float>& sampleScales,
                               cv::Mat& visualization);

  static void visualizeAllClusters(const std::vector<ssig::Cluster>& clusters,
                                   const std::vector<std::string>& mapSampleImagename,
                                   const std::vector<cv::Rect>& mapSamplePatch,
                                   std::vector<float>& sampleScales,
                                   const int cols,
                                   cv::Mat& visualization);

  static void visualizeAllClusters(const std::vector<ssig::Cluster>& clusters,
                                   const std::vector<std::string>& mapSampleImagename,
                                   const std::vector<cv::Rect>& mapSamplePatch,
                                   std::vector<float>& sampleScales,
                                   const std::string& outdir);
};

#endif

