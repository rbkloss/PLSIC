#include "reid.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <fstream>

#include <ml/pls_image_clustering.hpp>
#include <ml/pls_classifier.hpp>
#include <ml/oaa_classifier.hpp>

#include "Initializator.h"
#include "clustering.h"
#include <core/similarity_builder.hpp>


void Reid::extract(const std::string& filename,
                   const int cols,
                   cv::Mat_<float>& features) {
  std::ifstream inpfile(filename);
  float fvalue = 0.0f;

  for(int row = 0; row < 1264; ++row){
    cv::Mat_<float> feat(1, cols);
    feat = 0.0f;
    for(int i = 0; i < cols; ++i){
      inpfile >> fvalue;
      feat[0][i] = fvalue;
    }
    features.push_back(feat);
  }
}

void Reid::clusterPlsic(const cv::Mat_<float>& inp) const{
  cv::Mat_<float> cameraA = inp.rowRange(0, 632);
  cv::Mat_<float> cameraB = inp.rowRange(632, 1264);
  cv::Mat_<int> subset(1, 632);
  for(int i = 0; i < 632; ++i){
    subset[0][i] = i;
  }
  cv::randShuffle(subset, 3.0);
  cv::Mat_<float> train(316, 70);
  std::ofstream subsetFile("reid_subset.txt");
  for(int i = 0; i < 316; ++i){
    auto idx = subset[0][i];
    subsetFile << idx << std::endl;
    cameraB.row(idx).copyTo(train.row(i));
  }
  subsetFile.close();

  ssig::PLSClassifier pls;

  int nfactors, mValue, maxIt, maximumMergedPairs;
  std::string repType, simFunction, initializationMode;
  float mergeThresh, stdTreshold;
  bool respNorm, mergeConvergence;

  cv::FileStorage params("plsicParams.yml", cv::FileStorage::READ);
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

  auto discovery = Clustering::split(train);

  std::vector<ssig::Cluster> initialClustering;
  if(initializationMode == "random"){
    RandomInitializator init;
    initialClustering = init(train, discovery[0]);
  } else if(initializationMode == "kmeans"){
    KmeansInitializator init;
    initialClustering = init(train, discovery[0]);
  } else if(initializationMode == "split"){
    SplitInitializator init;
    initialClustering = init(train, discovery[0]);
  } else if(initializationMode == "splitjoin"){
    SplitJoinInitializator init;
    initialClustering = init(train, discovery[0]);
  } else if(initializationMode == "oss"){
    Ossigiltering init;
    RandomInitializator rnd;
    auto rndInit = rnd(train, discovery[0]);
    initialClustering = init(train, discovery, rndInit);
  }

  pls.setNumberOfFactors(nfactors);
  ssig::OAAClassifier oaa(pls);
  ssig::PLSImageClustering plsic(oaa, discovery, initialClustering);
  plsic.setClassifier(oaa);
  plsic.setMergeThreshold(mergeThresh);
  plsic.setDeviationThreshold(stdTreshold);
  plsic.setMValue(mValue);
  plsic.setMaxIterations(maxIt);
  plsic.setNormalizeResponses(respNorm);
  plsic.setMergeConvergence(mergeConvergence);
  plsic.setMaximumMergedPairs(maximumMergedPairs);

  if(repType == "centroids"){
    plsic.setClusterRepresentationType(ssig::ClusterRepresentationType::Centroids);
  } else if(repType == "responses"){
    plsic.setClusterRepresentationType(ssig::ClusterRepresentationType::ClustersResponses);
  }
  if(simFunction == "correlation"){
    plsic.setSimBuilder(ssig::SimilarityBuilder::correlationFunction);
  } else if(simFunction == "cosine"){
    plsic.setSimBuilder(ssig::SimilarityBuilder::cosineFunction);
  }


  std::sort(initialClustering.begin(), initialClustering.end());

  std::vector<ssig::Cluster> natVector = {{},{}};
  cv::Mat_<float> neg;
  plsic.addNaturalWorld(neg, natVector);
  plsic.setDiscoveryConfiguration(discovery);

  plsic.learn(train);

  auto clustering = plsic.getClustering();
  auto resps = plsic.getClustersResponses();

  std::ofstream ans("reid.txt");
  cv::Mat vis;
  int clusterOrder = -1;
  for(auto& cluster : clustering){
    ++clusterOrder;
    auto& resp = resps[clusterOrder];
    std::vector<int> ordering, orderedCluster;
    ssig::Util::sort(resp, resp.size(), ordering);
    std::reverse(ordering.begin(), ordering.end());
    ssig::Util::reorder(cluster, ordering, orderedCluster);
    cluster = orderedCluster;

    std::string path = "d:\\Docs\\Development\\DataSets\\Reid\\VIPeR\\b\\";
    visualize(cluster, path, vis);
    char visname[128];
    sprintf(visname, "reid_%03d.png", clusterOrder);
    cv::imwrite(visname, vis);
    for(auto& sample : cluster){
      ans << subset[0][sample] << " ";
    }
    ans << std::endl;
  }
  std::ofstream predFile("reid_prediction.txt");
  for(int i = 0; i < 316; ++i){
    auto feat = train.row(i);
    cv::Mat_<float> resp;
    plsic.predict(feat, resp);
    int max[2];
    cv::minMaxIdx(resp, nullptr, nullptr, nullptr, max);
    predFile << subset[0][i] << " " << max[1] << std::endl;
  }
}

void Reid::clusterKmeans(const cv::Mat_<float>& inp) const{
  cv::Mat_<float> cameraA = inp.rowRange(0, 631);
  cv::Mat_<float> cameraB = inp.rowRange(632, 1263);
  cv::Mat_<float> trainA = cameraA;

  ssig::Kmeans kmeans;

  int k;

  cv::FileStorage params("kmeansParams.yml", cv::FileStorage::READ);
  params["k"] >> k;
  kmeans.setK(k);
  kmeans.setFlags(cv::KMEANS_RANDOM_CENTERS);

  kmeans.learn(trainA);

  auto clustering = kmeans.getClustering();

  std::ofstream ans("reid.out");
  cv::Mat vis;
  int clusterOrder = -1;
  for(auto& cluster : clustering){
    std::string path = "d:\\Docs\\Development\\DataSets\\Reid\\VIPeR\\a\\";
    visualize(cluster, path, vis);
    char visname[128];
    sprintf(visname, "reid_%03d.png", ++clusterOrder);
    cv::imwrite(visname, vis);
    for(auto& sample : cluster){
      ans << sample << " ";
    }
    ans << std::endl;
  }
}

void Reid::visualize(const std::vector<int>& cluster,
                     const std::string& path,
                     cv::Mat& vis) const{
  //128x48
  const int len = static_cast<int>(cluster.size());
  vis.create(128, 48 * len, CV_8UC3);
  char imgname[8];

  cv::Mat block;
  int order = -1;
  for(auto& id : cluster){
    sprintf(imgname, "%03d.bmp", id + 1);
    block = cv::imread(path + imgname);
    int i = ++order * 48;
    int j = (order + 1) * 48;
    auto patch = vis.colRange(i, j);
    block.copyTo(patch);
  }
}

void Reid::execute() const{
  cv::Mat_<float> feats;
  extract("d:\\Docs\\Development\\DataSets\\Reid\\features.txt", 70, feats);
  clusterPlsic(feats);
  //clusterKmeans(feats);
}
