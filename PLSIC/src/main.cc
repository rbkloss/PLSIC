#include <string>
#include <ctime>

#include <ml/pls_classifier.hpp>
#include <ml/pls_image_clustering.hpp>
#include <ml/kmeans.hpp>

#include "reid.h"
#include "extraction.h"
#include "clustering.h"
#include "bagging_classifier.h"
#include <core/similarity_builder.hpp>

std::shared_ptr<ssig::PLSImageClustering> configurate(
  cv::FileStorage params);

int main(int argc, char** argv){
  cv::setUseOptimized(true);
  if(argc < 2){
    fprintf(stdout, "usage: PLISC.exe seed [extract, cluster, bag] [further options]\n");
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

  auto function = std::string(argv[2]);
 
  if(function == "reid"){
    Reid reid;
    reid.execute();
  } else if(function == "extract"){
    if(argc < 9){
      fprintf(stdout, "usage: PLISC.exe seed extract path\\to\\structure_file.yml\
                       path\\to\\save\\features \
                       windowWidth windowHeight \
                       strideX strideY \n");
      return 0;
    }
    auto stride = std::make_pair(std::stof(argv[7]), std::stof(argv[8]));
    auto windowSize = cv::Size(std::stoi(argv[5]),
                               std::stoi(argv[6]));
    Extraction::extractFeatures(argv[3],
                                argv[4],
                                windowSize,
                                stride
    );
  } else if(function == "cluster"){
    if(argc < 5){
      fprintf(stdout, "usage: PLISC.exe seed cluster\
                      [plsic,singh,kmeans] \
                      path\\to\\train \
                      [path\\to\\negatives] \
                      model_name \n");
      return 0;
    }
    printf("Clustering for Baggings\n");
    Clustering clustering;
    std::string method = argv[3];
    if(method == "plsic"){
      cv::FileStorage params("plsicParams.yml", cv::FileStorage::READ);
      std::string initmode;
      auto plsic = configurate(params);
      clustering.execute(*plsic, initmode, argv[4], argv[5]);
    } else if(method == "singh"){
      if(argc < 7){
        fprintf(stdout, "This option also needs the path to the negatives");
        return 0;
      }
      ssig::Singh singh;
      clustering.execute(singh, argv[4], argv[5], argv[6]);
    } else if(method == "kmeans"){
      cv::FileStorage params("kmeansParams.yml", cv::FileStorage::READ);
      ssig::Kmeans kmeans;
      int k;
      params["k"] >> k;
      kmeans.setK(k);
      clustering.execute(kmeans, argv[4], argv[5]);
    }

  } else if(function == "bag"){
    if(argc < 9){
      fprintf(stdout, "usage:\
                      PLISC.exe \
                      seed \
                      bag \
                      mode[singh, plsic, kmeans] \
                      path\\to\\attributes_file.yml \
                      path\\to\\train\\features path\\to\\train\\features \
                      path\\to\\train\\features path\\to\\test\\features \
                      path\\to\\clustering\\model.yml \
                      nfactors \n");
      return 0;
    }
    auto nfactors = std::stoi(argv[8]);
    auto mode = std::string(argv[3]);

    ssig::PLSClassifier pls;
    pls.setNumberOfFactors(nfactors);
    ssig::OAAClassifier oaa(pls);
    if(mode == "singh"){
      ssig::Singh singh;
      singh.setClassifier(oaa);
      singh.load(argv[7], "root");

      BaggingClassifier bagging;
      bagging.execute(argv[4],
                      argv[5],
                      argv[6],
                      singh,
                      oaa);
    } else if(mode == "plsic"){
      ssig::PLSImageClustering plsic(oaa, {}, {});
      plsic.load(argv[7], "root");

      BaggingClassifier bagging;
      bagging.execute(argv[4],
                      argv[5],
                      argv[6],
                      plsic,
                      oaa);
    } else if(mode == "kmeans"){
      ssig::Kmeans kmeans;
      kmeans.load(argv[7], "root");
      kmeans.setFlags(cv::KMEANS_RANDOM_CENTERS);
      kmeans.setPredicitonDistanceType(cv::NORM_L2);
      BaggingClassifier bagging;
      bagging.execute(argv[4],
                      argv[5],
                      argv[6],
                      kmeans,
                      oaa);
    }

  } else{
    printf("unrecognized option\n");
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
