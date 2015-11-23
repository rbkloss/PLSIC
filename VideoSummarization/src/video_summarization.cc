#include "video_summarization.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <random>

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ml/pls_classifier.hpp>
#include <ml/kmeans.hpp>
#include <core/util.hpp>
#include "Initializator.h"
#include <memory>
#include <ml/pls_image_clustering.hpp>
#include <core/similarity_builder.hpp>

std::vector<std::vector<int>> split(cv::Mat_<float>& inp){
  std::vector<std::vector<int>> ans;
  ans.resize(2);
  const int nsamples = inp.rows;
  const int halflen = nsamples / 2;

  std::vector<int> ids(nsamples, 0);
  for(int i = 0; i < nsamples; ++i){
    ids[i] = i;
  }

  std::shuffle(ids.begin(), ids.end(), std::default_random_engine());
  for(int i = 0; i < halflen; ++i){
    ans[0].push_back(ids[i]);

  }
  for(int i = halflen + 1; i < nsamples; ++i){
    ans[1].push_back(ids[i]);
  }

  return ans;
}

std::shared_ptr<ssig::PLSImageClustering> VideoSummarization::configurePlsic(const std::string& plsicConfig,
                                                                            const std::string& initConfig,
                                                                            const int videoId,
                                                                            cv::Mat_<float>& features){
  int nfactors, mValue, maxIt, maximumMergedPairs;
  std::string repType, simFunction, initializationMode;
  float mergeThresh, stdTreshold;
  bool respNorm, mergeConvergence;
  cv::FileStorage params(plsicConfig, cv::FileStorage::READ);
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

  auto discovery = split(features);
  std::vector<ssig::Cluster> initialClustering;
  if(initializationMode == "random"){
    RandomInitializator init;
    initialClustering = init(features, discovery[0]);
  } else if(initializationMode == "kmeans"){
    KmeansInitializator init;
    initialClustering = init(features, discovery[0]);
  } else if(initializationMode == "split"){
    SplitInitializator init;
    initialClustering = init(features, discovery[0]);
  } else if(initializationMode == "splitjoin"){
    SplitJoinInitializator init;
    initialClustering = init(features, discovery[0]);
  } else if(initializationMode == "avila"){

    cv::Mat_<float> hsvHistograms;
    for(int i = 0; i < videoFrames_.size(); i += frameSampling_){
      cv::Mat img;
      cv::Mat_<float> hist;
      auto& frame = videoFrames_[i];
      if(!frame.empty()){
        cv::cvtColor(frame, img, CV_RGB2HSV);
        int channels[] = {0};
        int histSize[] = {16};
        float hranges[] = {0, 180};
        const float* ranges[] = {hranges};
        cv::calcHist(&img, 1, channels, cv::Mat(), hist, 1, histSize, ranges);
        cv::transpose(hist, hist);
      } else
        hist = cv::Mat_<float>::zeros(1, 16);
      hsvHistograms.push_back(hist);
    }
    AvilaInitializator init;
    init(hsvHistograms, discovery[0], initialClustering);
    //###################
    //features = hsvHistograms;

  } else if(initializationMode == "oss"){
    Ossigiltering init;
    RandomInitializator rnd;
    auto rndInit = rnd(features, discovery[0]);
    initialClustering = init(features, discovery, rndInit);
  }

  pls.setNumberOfFactors(nfactors);
  ssig::OAAClassifier oaa(pls);
  auto plsic = std::make_shared<ssig::PLSImageClustering>(oaa, discovery, initialClustering);
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
    plsic->setSimBuilder([](cv::Mat_<float>& x, cv::Mat_<float>& y){
      auto comp = cv::compareHist(x, y, cv::HISTCMP_CORREL);
      return static_cast<float>(comp);
    });
  } else if(simFunction == "cosine"){
    plsic->setSimBuilder(ssig::SimilarityBuilder::cosineFunction);
  }

  if(viewBegin_){
    char videoname[512];
    sprintf(videoname, "%sv%d.mpg", videoPath, videoId);
    std::sort(initialClustering.begin(), initialClustering.end());
    std::vector<cv::Mat> clustersVis;
    visualizeClusters(initialClustering, clustersVis);

    int COL = 0;
    for(auto& img : clustersVis){
      if(img.cols > COL)
        COL = img.cols;
    }

    cv::Mat vis;
    for(auto& img : clustersVis){
      cv::Mat newmat(img.rows, COL, CV_8UC3);
      auto roi = newmat(cv::Rect(0, 0, img.cols, img.rows));
      img.copyTo(roi);
      if(vis.empty())
        vis = newmat;
      else
        cv::vconcat(vis, newmat, vis);
    }
    cv::imwrite("begin" + std::to_string(videoId) + ".jpg", vis);
    vis.release();
  }
  std::vector<ssig::Cluster> natVector = {{},{}};
  cv::Mat_<float> neg;
  plsic->addNaturalWorld(neg, natVector);
  return plsic;
}

std::vector<ssig::Cluster> VideoSummarization::pickKeyframes(const cv::Mat& feats,
  const std::vector<ssig::Cluster>& clusters,
  const std::vector<std::vector<float>>& clustersResponses){
  std::vector<ssig::Cluster> frames;
  //Responses:
  if (responses_) {
    for (int i = 0; i < clusters.size(); ++i) {
      auto cluster = clusters[i];
      float maxResp = -FLT_MAX;
      int bestId = -1;
      for (int j = 0; j < cluster.size(); ++j) {
        const auto resp = clustersResponses[i][j];
        if (resp >= maxResp) {
          maxResp = resp;
          bestId = cluster[j];
        }
      }
      frames.push_back({ bestId });
    }
  }else{//Centroids:
    for (int i = 0; i < clusters.size(); ++i) {
      auto cluster = clusters[i];
      cv::Mat_<float> centroid(1, feats.cols, 0.0f);
      float bestResp = FLT_MAX;
      int bestId = -1;
      for (int j = 0; j < cluster.size(); ++j){
        auto feat = feats.row(cluster[j]);
        centroid += feat;
      }
      centroid = centroid / static_cast<double>(cluster.size());

      for (int j = 0; j < cluster.size(); ++j) {
        auto feat = feats.row(cluster[j]);
        auto resp = static_cast<float>(cv::norm(feat - centroid));
        if (resp <= bestResp) {
          bestResp = resp;
          bestId = cluster[j];
        }
      }
      frames.push_back({ bestId });
    }
  }
  //
  return frames;
}

void VideoSummarization::readFeatures(const std::string& filename,
                                      std::vector<int>& frameIds,
                                      cv::Mat_<float>& features){
  frameIds.clear();
  int nFrames, matOrder, matPerFrame;
  std::ifstream input(filename);
  std::string line;
  std::getline(input, line);
  std::stringstream ss(line);

  ss >> nFrames;
  nFrames = static_cast<int>(videoFrames_.size());
  ss >> matOrder;
  ss >> matPerFrame;

  const int dimensions = matOrder * matOrder * matPerFrame;
  int i = 0;
  while(i < nFrames){
    cv::Mat_<float> feat(1, dimensions);
    int j = 0;
    while(j < dimensions){
      float value;
      input >> value;
      feat[0][j] = value;
      ++j;
    }
    if(i % frameSampling_ == 0){
      frameIds.push_back(i);
      features.push_back(feat);
    }
    ++i;
  }

}

std::vector<ssig::Cluster> VideoSummarization::cluster(cv::Mat_<float>& inp,
                                                      ssig::Clustering& clusteringMethod){
  clusteringMethod.learn(inp);
  return clusteringMethod.getClustering();
}

std::vector<ssig::Cluster> VideoSummarization::cluster(const int videoid,
                                                      cv::Mat_<float>& inp,
                                                      ssig::ClassifierClustering& clusteringMethod){
  fprintf(stdout, "Method setup\n");
  clusteringMethod.setup(inp);
  fprintf(stdout, "Setup Over\n");

  int it = 0;
  do{
    fprintf(stdout, "Iterating\n");
    if(viewPartial_){
      auto state = clusteringMethod.getClustering();
      auto resps = clusteringMethod.getClustersResponses();
      for(int i = 0; i < state.size(); ++i){
        auto& cluster = state[i];
        auto& resp = resps[i];
        std::vector<int> ordering;
        ssig::Util::sort(resp, resp.size(), ordering);
        std::reverse(ordering.begin(), ordering.end());
        ssig::Cluster sortedCluster;
        ssig::Util::reorder(cluster, ordering, sortedCluster);
        cluster = sortedCluster;
      }

      std::vector<cv::Mat> clustersVis;
      visualizeClusters(state, clustersVis);
      char number[10];
      cv::Mat vis;
      for(auto& img : clustersVis){
        if(vis.empty())
          vis = img;
        else{
          cv::Mat newmat(img.rows, vis.cols, CV_8UC3);
          auto roi = newmat(cv::Rect(0, 0, img.cols, img.rows));
          img.copyTo(roi);
          cv::vconcat(vis, newmat, vis);
        }
      }
      sprintf(number, "%05d", ++it);
      cv::imwrite("Clustering_v" + std::to_string(videoid) + "_" + std::string(number) + ".jpg", vis);
    }
  } while(!clusteringMethod.iterate());

  auto state = clusteringMethod.getClustering();

  if(viewFinal_){
    auto resps = clusteringMethod.getClustersResponses();
    for(int i = 0; i < state.size(); ++i){
      auto& cluster = state[i];
      auto& resp = resps[i];
      std::vector<int> ordering;
      ssig::Util::sort(resp, resp.size(), ordering);
      std::reverse(ordering.begin(), ordering.end());
      ssig::Cluster sortedCluster;
      ssig::Util::reorder(cluster, ordering, sortedCluster);
      cluster = sortedCluster;
    }
    std::vector<cv::Mat> clustersVis;
    visualizeClusters(state, clustersVis);
    char number[10];
    cv::Mat vis;
    for(auto& img : clustersVis){
      if(vis.empty())
        vis = img;
      else{
        cv::Mat newmat(img.rows, vis.cols, CV_8UC3);
        auto roi = newmat(cv::Rect(0, 0, img.cols, img.rows));
        img.copyTo(roi);
        cv::vconcat(vis, newmat, vis);
      }
    }
    sprintf(number, "%05d", ++it);
    cv::imwrite("Clustering_v" + std::to_string(videoid) + "_final" + ".jpg", vis);
  }
  return state;
}

void VideoSummarization::visualizeClusters(
  const std::vector<ssig::Cluster>& clustering,
  std::vector<cv::Mat>& visualization){
  const int width = videoFrames_[0].cols;
  const int height = videoFrames_[0].rows;
  std::vector<cv::Mat> ans;
  for(auto& cluster : clustering){
    const int len = static_cast<int>(cluster.size());
    cv::Mat vis;
    vis.create(height, width * len, CV_8UC3);
    int j = 0;
    for(auto& id : cluster){
      auto patch = vis.colRange(j * width, (j + 1) * width);
      int frameId = frameIds_[id];
      videoFrames_[frameId].copyTo(patch);
      ++j;
    }
    ans.push_back(vis);
  }
  visualization = ans;
}

void VideoSummarization::loadVideo(const std::string& videoname,
                                   std::vector<cv::Mat>& videoFrames){
  cv::VideoCapture capture(videoname);

  if(!capture.isOpened()){
    std::cout << "Error opening video!" << std::endl;
    exit(1);
  }

  int totalFrames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
  capture.set(cv::CAP_PROP_CONVERT_RGB, 1);

  videoFrames.resize(totalFrames);
  cv::Mat frame;
  for(int i = 0; i < totalFrames; ++i){
    if(!capture.read(frame)){
      fprintf(stderr, "cv::VideoCapture failed file [%s]-[%d]of[%d]\n", videoname.c_str(), i, totalFrames);
    }
    frame.copyTo(videoFrames[i]);
  }
}

VideoSummarization::VideoSummarization(const int sampling, const bool selectKeyframesResponses) :
frameSampling_(sampling), responses_(selectKeyframesResponses) {}

void VideoSummarization::executePlsic(const std::string& plsicConfig,
                                      const std::string& initConfig){
  std::ifstream kFile("keyframes.txt");
  if(!kFile.is_open()){
    printf("could not locate key frame file\n");
    exit(-1);
  }
  std::vector<int> kVector;
  while(kFile){
    int k;
    kFile >> k;
    kVector.push_back(k);
  }
  kFile.close();
  for(int videoId = 21; videoId <= 70; ++videoId){
    char featfilename[512], videoname[512];
    sprintf(featfilename, "%s/v%d.txt", featPath, videoId);
    sprintf(videoname, "%sv%d.mpg", videoPath, videoId);
    printf("Analyzing video[%d]\n", videoId
    );
    videoFrames_.clear();
    loadVideo(videoname, videoFrames_);
    cv::Mat_<float> feats;

    readFeatures(featfilename, frameIds_, feats);

    auto clusteringMethod = configurePlsic(plsicConfig, initConfig, videoId, feats);

    clusteringMethod->setK(kVector[videoId - 21]);

    cluster(videoId, feats, *clusteringMethod);

    auto clusters = clusteringMethod->getClustering();
    auto responses = clusteringMethod->getClustersResponses();
    auto frames = pickKeyframes(feats, clusters, responses);

    std::vector<cv::Mat> finalVis;
    visualizeClusters(frames, finalVis);
    int i = -1;
    for(auto& frameMat : finalVis){
      char framename[512];
      auto frameId = frameIds_[frames[++i][0]];
      sprintf(framename, "%s/plsic/v%d/Frame%05d.jpg", summaryPath, videoId, frameId);
      cv::imwrite(framename, frameMat);
    }
  }
}

void VideoSummarization::executeKmeans(const std::string& kmeansConfig,
                                       const std::string& initConfig){
  ///Setup
  std::ifstream kFile("keyframes.txt");
  std::vector<int> kVector;
  while(kFile){
    int k;
    kFile >> k;
    kVector.push_back(k);
  }
  kFile.close();
  ///
  for(int videoId = 21; videoId < 71; ++videoId){
    ssig::Kmeans kmeans;
    kmeans.setFlags(cv::KMEANS_RANDOM_CENTERS);
    kmeans.setNAttempts(1);
    kmeans.setPredicitonDistanceType(cv::NORM_L2);
    kmeans.setK(kVector[videoId - 21]);
    char featfilename[512], videoname[512];
    sprintf(featfilename, "%s/v%d.txt", featPath, videoId);
    sprintf(videoname, "%s/v%d.mpg", videoPath, videoId);
    videoFrames_.clear();
    loadVideo(videoname, videoFrames_);
    cv::Mat_<float> feats;
    readFeatures(featfilename, frameIds_, feats);
    auto clusters = cluster(feats, kmeans);
    cv::Mat_<float> centroids;
    kmeans.getCentroids(centroids);
    std::vector<ssig::Cluster> frames;
    for(int i = 0; i < clusters.size(); ++i){
      auto cluster = clusters[i];
      float maxResp = -FLT_MAX;
      int bestId = -1;
      for(int j = 0; j < cluster.size(); ++j){
        int id = cluster[j];
        const auto resp = static_cast<float>(-1 * cv::norm(feats.row(id) - centroids.row(i)));
        if(resp > maxResp){
          maxResp = resp;
          bestId = cluster[j];
        }
      }
      frames.push_back({bestId});
    }

    std::vector<cv::Mat> finalVis;
    visualizeClusters(frames, finalVis);
    int i = -1;
    for(auto& frameMat : finalVis){
      char framename[512];
      sprintf(framename, "%s/kmeans/v%d/Frame%05d.jpg", summaryPath, videoId, frames[++i][0]);
      cv::imwrite(framename, frameMat);
    }
  }
}
