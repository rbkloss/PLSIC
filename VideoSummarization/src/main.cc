#include <string>
#include <ctime>
#include <memory>

#include <ml/pls_classifier.hpp>
#include <ml/pls_image_clustering.hpp>
#include <ml/kmeans.hpp>

#include <opencv2/core.hpp>

#include "video_summarization.h"
#include "Initializator.h"
#include <core/similarity_builder.hpp>


std::shared_ptr<ssig::PLSImageClustering> configurate(
  cv::FileStorage params);

int main(int argc, char** argv){
  cv::setUseOptimized(true);
  if(argc < 6){
    fprintf(stdout, "usage: PLSIC.exe seed \
                     [plsic/kmeans]\
                     methodParams.yml \
                     initializationParams.yml \
                     frameSampling \
                     UseResponses[yes\no]\n");
    return 0;
  }
  long seed;
  if(strcmp(argv[1], "time") == 0){
    seed = static_cast<long>(time(nullptr));
  } else{
    seed = std::stol(argv[1]);
  }
  printf("Seed set to [%ld]\n", seed);
  if(seed){
    srand(static_cast<unsigned>(seed));
    cv::theRNG().state = seed;
  }

  bool respOrCent = false;
  if(strcmp(argv[6], "yes") == 0){
    respOrCent = true;
  } else if(strcmp(argv[6], "no") == 0)
    respOrCent = false;

  VideoSummarization video(std::stoi(argv[5]), respOrCent);
  std::string method = argv[2];
  if(method == "plsic"){
    video.executePlsic(argv[3], argv[4]);
  } else if(method == "kmeans"){
    ssig::Kmeans kmeans;
    video.executeKmeans(argv[3], argv[4]);
  }

  return 0;
}

std::shared_ptr<ssig::PLSImageClustering> configurate(
  cv::FileStorage params){
  int nfactors, mValue, maxIt, maximumMergedPairs;
  std::string repType, simFunction, initializationMode;
  float mergeThresh, stdTreshold;
  bool respNorm, mergeConvergence;

  ssig::PLSClassifier pls;
  params["nfactors"] >> nfactors;
  params["mValue"] >> mValue;
  params["maxIt"] >> maxIt;
  params["representation"] >> repType;
  params["similarity"] >> simFunction;
  params["mergeThreshold"] >> mergeThresh;
  params["initialization"] >> initializationMode;
  params["stdTreshold"] >> stdTreshold;
  params["respNorm"] >> respNorm;
  params["mergeConvergence"] >> mergeConvergence;
  params["maximumMergedPairs"] >> maximumMergedPairs;

  pls.setNumberOfFactors(nfactors);
  ssig::OAAClassifier oaa(pls);
  auto plsic = std::make_shared<ssig::PLSImageClustering>(oaa, std::vector<std::vector<int>>(), std::vector<ssig::Cluster>());
  plsic->setMergeThreshold(mergeThresh);
  plsic->setDeviationThreshold(stdTreshold);
  plsic->setMValue(mValue);
  plsic->setMaxIterations(maxIt);
  plsic->setNormalizeResponses(respNorm);
  plsic->setMergeConvergence(mergeConvergence);
  plsic->setMaximumMergedPairs(maximumMergedPairs);

  if(repType == "centroids"){
    plsic->setClusterRepresentationType(ssig::ClusterRepresentationType::Centroids);
  } else if(repType == "responses"){
    plsic->setClusterRepresentationType(ssig::ClusterRepresentationType::ClustersResponses);
  }
  if(simFunction == "correlation"){
    plsic->setSimBuilder(ssig::SimilarityBuilder::correlationFunction);
  } else if(simFunction == "cosine"){
    plsic->setSimBuilder(ssig::SimilarityBuilder::cosineFunction);
  }

  return plsic;
}
