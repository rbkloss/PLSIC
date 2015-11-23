#ifndef _BAGGING_CLASSIFIER_H_
#define _BAGGING_CLASSIFIER_H_

#include <ml/oaa_classifier.hpp>
#include <ml/classifier_clustering.hpp>
#include <ml/clustering.hpp>
#include <memory>

class BaggingClassifier{

  void makeBaggings(const std::string& attributesFilename,
                    const std::string& path,
                    std::unordered_map<std::string, cv::Mat_<float>>& features,
                    std::vector<std::string>& labels,
                    ssig::Clustering& baggingBuilder) const;
  void makeBaggings(const std::string& attributesFilename,
                    const std::string& path,
                    cv::Mat_<float>& features,
                    std::vector<std::string>& filenames,
                    std::vector<std::string>& labels,
                    ssig::Clustering& baggingBuilder) const;

  void trainClassifier(const std::vector<std::string>& labelOrdering,
                       const std::unordered_map<std::string, cv::Mat_<float>>& baggings,
                       ssig::OAAClassifier& classifier) const;
  void test(const std::string& outDir,
            const std::vector<std::string>& labelOrdering,
            const std::vector<std::string>& filenames,
            const cv::Mat_<float>& baggings,
            const ssig::OAAClassifier& classifier)const;
public:
  void execute(const std::string& attributesFilename,
               const std::string& trainPath,
               const std::string& testPath,
               ssig::Clustering& baggingBuilder,
               ssig::OAAClassifier& classifier) const;

};

#endif
