#include "bagging_classifier.h"

#include "extraction.h"

#include <opencv2/core.hpp>
#include <set>
#include <fstream>

void BaggingClassifier::execute(const std::string& attributesFilename,
                                const std::string& trainPath,
                                const std::string& testPath,
                                ssig::Clustering& baggingBuilder,
                                ssig::OAAClassifier& classifier) const{
  std::vector<std::string> labels;
  std::unordered_map<std::string, cv::Mat_<float>> trainBaggings;
  makeBaggings(attributesFilename,
               trainPath,
               trainBaggings,
               labels,
               baggingBuilder);

  trainClassifier(labels, trainBaggings, classifier);

  cv::Mat_<float> testBaggings;
  std::vector<std::string> filenames;
  makeBaggings(attributesFilename,
               testPath,
               testBaggings,
               filenames,
               labels,
               baggingBuilder);

  test(".//",
       labels,
       filenames,
       testBaggings,
       classifier);
}


void BaggingClassifier::makeBaggings(const std::string& attributesFilename,
                                     const std::string& path,
                                     std::unordered_map<std::string, cv::Mat_<float>>& features,
                                     std::vector<std::string>& labels,
                                     ssig::Clustering& baggingBuilder) const{
  std::set<std::string> labelsSet;
  std::vector<cv::String> listing;
  cv::FileStorage attributes(attributesFilename, cv::FileStorage::READ);
  cv::glob(path + "*.yml", listing);
  std::string prefix;
  attributes["prefix"] >> prefix;
  for(auto& absFilename : listing){
    std::string rootname;
    auto pos = absFilename.find_last_of("\\");
    std::string filename = absFilename.substr(pos + 1);
    pos = filename.find_last_of(".yml");
    rootname = filename.substr(0, pos - 3);

    auto fileNode = attributes[prefix + rootname];
    std::vector<std::string> classes;
    auto classesNode = fileNode["classes"];
    auto it = classesNode.begin();
    for(; it != classesNode.end(); ++it){
      std::string label;
      *it >> label;
      classes.push_back(label);
    }

    std::string name = absFilename;
    cv::Mat_<float> fileFeatures;
    std::string imagename;
    std::vector<cv::Rect> patches;
    std::vector<float> sampleScales;
    Extraction::readFeatures(name, fileFeatures, imagename, patches, sampleScales);

    cv::checkRange(fileFeatures, false);

    cv::Mat_<float> baggings;
    baggings = 0;
    for(int r = 0; r < fileFeatures.rows; ++r){
      cv::Mat_<float> bagging;
      auto feat = fileFeatures.row(r);
      baggingBuilder.predict(feat, bagging);
      double min, max;
      int maxIdx[2];
      //baggings.at<float>(idxes) = bagging.at<float>(idxes);
      cv::minMaxIdx(bagging, &min, &max, nullptr, maxIdx);
      baggings[maxIdx[0]][maxIdx[1]] += 1;
    }
    baggings = baggings / (cv::norm(baggings) + FLT_EPSILON);

    for(auto& label : classes){
      features[label].push_back(baggings);
      labelsSet.insert(label);
    }
  }
  labels.assign(labelsSet.begin(), labelsSet.end());
}

void BaggingClassifier::makeBaggings(const std::string& attributesFilename,
                                     const std::string& path,
                                     cv::Mat_<float>& features,
                                     std::vector<std::string>& filenames,
                                     std::vector<std::string>& labels,
                                     ssig::Clustering& baggingBuilder) const{
  std::set<std::string> labelsSet;
  std::vector<cv::String> listing;
  cv::FileStorage attributes(attributesFilename, cv::FileStorage::READ);
  cv::glob(path + "*.yml", listing);
  std::string prefix;
  attributes["prefix"] >> prefix;
  for(auto& absFilename : listing){
    std::string rootname;
    auto pos = absFilename.find_last_of("\\");
    std::string filename = absFilename.substr(pos + 1);
    pos = filename.find_last_of(".yml");
    rootname = filename.substr(0, pos - 3);
    filenames.push_back(rootname);

    auto fileNode = attributes[prefix + rootname];
    std::vector<std::string> classes;
    auto classesNode = fileNode["classes"];
    auto it = classesNode.begin();
    for(; it != classesNode.end(); ++it){
      std::string label;
      *it >> label;
      classes.push_back(label);
    }

    std::string name = absFilename;
    cv::Mat_<float> fileFeatures;
    std::string imagename;
    std::vector<cv::Rect> patches;
    std::vector<float> sampleScales;
    Extraction::readFeatures(name, fileFeatures, imagename, patches, sampleScales);

    cv::Mat_<float> baggings;
    for(int r = 0; r < fileFeatures.rows; ++r){
      cv::Mat_<float> bagging;
      auto feat = fileFeatures.row(r);
      baggingBuilder.predict(feat, bagging);
      double min, max;
      int idxes[2];
      cv::minMaxIdx(bagging, &min, &max, nullptr, idxes);
      //baggings.at<float>(idxes) = bagging.at<float>(idxes);
      baggings.at<float>(idxes) += 1;

    }
    baggings = baggings / (cv::norm(baggings) + FLT_EPSILON);
    features.push_back(baggings);

    for(auto& label : classes){
      labelsSet.insert(label);
    }
  }
  labels.assign(labelsSet.begin(), labelsSet.end());
}

void BaggingClassifier::trainClassifier(const std::vector<std::string>& labelOrdering,
                                        const std::unordered_map<std::string, cv::Mat_<float>>& baggings,
                                        ssig::OAAClassifier& classifier) const{

  cv::Mat_<int> labelsMat;
  cv::Mat_<float> inp;
  for(size_t l = 0; l < labelOrdering.size(); ++l){
    auto label = labelOrdering[l];
    auto bagging = baggings.at(label);
    inp.push_back(bagging);
    cv::Mat_<int> m(bagging.rows, 1, static_cast<int>(l));
    labelsMat.push_back(m);
  }
  classifier.learn(inp, labelsMat);
}

void BaggingClassifier::test(const std::string& outDir,
                             const std::vector<std::string>& labelsVector,
                             const std::vector<std::string>& filenames,
                             const cv::Mat_<float>& baggings,
                             const ssig::OAAClassifier& classifier) const{
  cv::Mat_<float> responseMatrix;
  auto labelOrdering = classifier.getLabelsOrdering();
  auto feats = baggings;
  cv::Mat_<float> responses;
  classifier.predict(feats, responses);
  responseMatrix.push_back(responses);


  for(int c = 0; c < static_cast<int>(labelsVector.size()); ++c){
    std::string label = labelsVector[c];
    std::string outname = outDir + "comp1_cls_test_" + label + ".txt";
    std::ofstream responseFile(outname);
    int idx = labelOrdering.at(c);
    for(int r = 0; r < responseMatrix.rows; ++r){
      auto response = responseMatrix[r][idx];
      responseFile << filenames.at(r) + " " + std::to_string(response) + "\n";
    }
    responseFile.close();
  }
}
