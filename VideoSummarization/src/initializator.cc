#include "Initializator.h"
#include <unordered_map>
#include <set>
#include <ml/pls_classifier.hpp>
#include <opencv2/imgproc.hpp>
#include <ml/oaa_classifier.hpp>

std::vector<ssig::Cluster> RandomInitializator::operator()(const cv::Mat_<float>& inp, const std::vector<int>& assignmentSet) const{
  //RANDOM init
  const int len = static_cast<int>(assignmentSet.size()) / 4;
  const int nsamples = static_cast<int>(assignmentSet.size());
  std::vector<ssig::Cluster> ans(len, {});
  cv::Mat_<int> randArray(1, nsamples);
  for(int i = 0; i < nsamples; ++i){
    randArray[0][i] = i;
  }
  cv::randShuffle(randArray, 5);

  for(int i = 0; i < len; ++i){
    ans[i].push_back(assignmentSet[randArray[0][i]]);
  }

  return ans;
}

std::vector<ssig::Cluster> KmeansInitializator::operator()(const cv::Mat_<float>& inp, const std::vector<int>& assignmentSet) const{
  std::vector<ssig::Cluster> ans;
  cv::Mat_<float> assignmentInput;
  for(auto id : assignmentSet){
    assignmentInput.push_back(inp.row(id));
  }
  const int K = static_cast<int>(assignmentSet.size() / 4);

  cv::Mat centroids, lab;
  cv::TermCriteria term;
  term.type = term.EPS | term.MAX_ITER;
  term.epsilon = 0.001;
  term.maxCount = static_cast<int>(1e4);
  cv::kmeans(assignmentInput, K, lab, term, 1, cv::KMEANS_RANDOM_CENTERS, centroids);

  std::unordered_map<int, ssig::Cluster> mapLabelClusters;
  cv::Mat_<int> labels = lab;
  for(int r = 0; r < labels.rows; ++r){
    int label = labels[r][0];
    mapLabelClusters[label].push_back(assignmentSet[r]);
  }
  for(auto& it : mapLabelClusters){
    auto cluster = it.second;
    if(cluster.size() > 3){
      float bestResp = FLT_MAX;
      int bestId = 0;
      auto centroid = centroids.row(it.first);
      for(auto id : cluster){
        cv::Mat_<float> feat = inp.row(id);
        float resp = static_cast<float>(cv::norm(centroid - feat));
        if(resp < bestResp){
          bestResp = resp;
          bestId = id;
        }
      }
      ans.push_back({bestId});
    }
  }

  return ans;
}

std::vector<ssig::Cluster> SplitInitializator::operator()(const cv::Mat_<float>& inp, const std::vector<int>& assignmentSet) const{
  auto clustering = assignmentSet;
  cv::FileStorage params("initializationParams.yml", cv::FileStorage::READ);
  if(!params.isOpened())exit(-1);
  float cosineThreshold, manhattanThreshold;
  params["cosineThreshold"] >> cosineThreshold;
  params["manhattanThreshold"] >> manhattanThreshold;
  std::vector<int> removables;
  const int len = static_cast<int>(clustering.size());
  for(int i = 0; i < len; ++i){
    int c1 = clustering[i];
    auto m1 = inp.row(c1);
    auto norm1 = cv::norm(m1);
    for(int j = i + 1; j < len; ++j){
      int c2 = clustering[j];
      auto m2 = inp.row(c2);
      auto norm2 = cv::norm(m2);
      auto dotSimilarity = static_cast<float>(cv::abs(m1.dot(m2)) / (norm1 * norm2));
      auto manhattanDistance = cv::norm(m1 - m2, cv::NORM_L1);
      if(dotSimilarity >= cosineThreshold || manhattanDistance <= manhattanThreshold){
        removables.push_back(j);
      }
    }
  }

  std::vector<ssig::Cluster> ans;
  int idx = 0;
  std::sort(removables.begin(), removables.end());
  auto it = std::unique(removables.begin(), removables.end());
  removables.resize(std::distance(removables.begin(), it));
  for(int i = 0; i < clustering.size(); ++i){
    if(i != removables[idx]){
      ans.push_back({clustering[i]});
    } else{
      ++idx;
    }
    if(idx >= removables.size())
      break;
  }

  return ans;
}

std::vector<ssig::Cluster> SplitJoinInitializator::operator()(const cv::Mat_<float>& inp, const std::vector<int>& assignmentSet) const{
  SplitInitializator spliter;
  auto initialPoints = spliter(inp, assignmentSet);
  const int len = static_cast<int>(initialPoints.size());
  std::set<int> added;
  for(auto& i : initialPoints)
    added.insert(i.front());

  for(int i = 0; i < len; ++i){
    auto m1 = inp.row(initialPoints[i][0]);
    auto norm1 = cv::norm(m1);
    for(int j = 0; j < inp.rows; ++j){
      if(added.find(j) == added.end()){
        auto m2 = inp.row(j);
        auto norm2 = cv::norm(m2);

        auto dotSimilarity = cv::abs(m1.dot(m2) / (norm1 * norm2));
        auto manhattanDistance = cv::norm(m1 - m2, cv::NORM_L1);
        float energyThresh = m1.cols * 0.01f;
        if(dotSimilarity >= 0.99 || manhattanDistance <= energyThresh){
          initialPoints[i].push_back(j);
          added.insert(j);
        }
      }
    }
  }
  return initialPoints;
}

std::vector<ssig::Cluster> SinghInitializator::operator()(const cv::Mat_<float>& inp, const std::vector<int>& assignmentSet) const{
  std::vector<ssig::Cluster> ans;
  cv::Mat_<float> assignmentInput;
  for(auto id : assignmentSet){
    assignmentInput.push_back(inp.row(id));
  }
  const int K = static_cast<int>(assignmentSet.size() / 4);

  cv::Mat centroids, lab;
  cv::TermCriteria term;
  term.type = term.EPS | term.MAX_ITER;
  term.epsilon = 0.001;
  term.maxCount = static_cast<int>(1e4);
  cv::kmeans(assignmentInput, K, lab, term, 1, cv::KMEANS_RANDOM_CENTERS, centroids);

  std::unordered_map<int, ssig::Cluster> mapLabelClusters;
  cv::Mat_<int> labels = lab;
  for(int r = 0; r < labels.rows; ++r){
    int label = labels[r][0];
    mapLabelClusters[label].push_back(assignmentSet[r]);
  }
  for(auto& it : mapLabelClusters){
    auto cluster = it.second;
    if(cluster.size() > 3){
      ans.push_back(cluster);
    }
  }

  return ans;
}

/**
*/
std::vector<ssig::Cluster> Ossigiltering::operator()(const cv::Mat_<float>& inp, const std::vector<std::vector<int>>& discovery,
                                                   const std::vector<ssig::Cluster>& initial){
  cv::FileStorage params("initializationParams.yml", cv::FileStorage::READ);
  if(!params.isOpened())exit(-1);
  int negativeLen, ossigactors;
  float ossThreshold;
  params["ossNegativeLen"] >> negativeLen;
  params["ossThreshold"] >> ossThreshold;
  params["ossNFactors"] >> ossigactors;
  const int len = static_cast<int>(initial.size());
  const int negSetLen = static_cast<int>(discovery[1].size());
  cv::Mat_<int> shuffled(negSetLen, 1, 0);
  for(int j = 0; j < negSetLen; ++j){
    shuffled[j][0] = j;
  }
  cv::Mat_<float> negMat;
  cv::Mat_<int> labels(1 + negativeLen, 1);
  labels[0][0] = 1;
  cv::randShuffle(shuffled, 5.0);
  for(int j = 0; j < negativeLen; ++j){
    int idx = discovery[1][shuffled[j][0]];
    negMat.push_back(inp.row(idx));
    labels[j + 1][0] = j + 1;
  }
  //oss's training
  std::vector<ssig::OAAClassifier*> ossClassifiers(initial.size(), nullptr);
  for(int i = 0; i < len; ++i){
    ssig::PLSClassifier pls;
    pls.setNumberOfFactors(ossigactors);
    ossClassifiers[i] = new ssig::OAAClassifier(pls);
    auto classifier = ossClassifiers[i];

    cv::Mat_<float> xMat;
    xMat.push_back(inp.row(initial[i][0]));
    xMat.push_back(negMat);
    classifier->learn(xMat, labels);
  }

  std::set<int> removables;
  for(int i = 0; i < len; ++i){
    auto featI = inp.row(initial[i][0]);
    cv::Mat_<float> respIJ, respJI;
    for(int j = i + 1; j < len; ++j){
      auto featJ = inp.row(initial[j][0]);
      ossClassifiers[i]->predict(featJ, respIJ);
      ossClassifiers[j]->predict(featI, respJI);

      auto comparison = cv::compareHist(respIJ, respJI, CV_COMP_CORREL);
      if(comparison > ossThreshold){
        removables.insert(i);
        break;
      }
      /* auto labIdx = ossClassifiers[i]->getLabelsOrdering();
      auto idxI = labIdx[1];
      labIdx = ossClassifiers[j]->getLabelsOrdering();
      auto idxJ = labIdx[1];
      if(respIJ[0][idxI] > ossThreshold){
        if(respJI[0][idxJ] > ossThreshold){
          removables.insert(i);
          break;
        }
      }*/
    }
  }

  std::vector<ssig::Cluster> resp;
  for(int i = 0; i < len; ++i){
    auto id = initial[i];
    if(removables.find(i) == removables.end()){//if id is not in removables
      resp.push_back(id);
    }
  }


  for(auto& classifier : ossClassifiers)
    delete classifier;
  return resp;
}

void AvilaInitializator::operator()(const cv::Mat_<float>& features,
  const std::vector<int>& assignmentSet,
  std::vector<ssig::Cluster>& initialClustering) const{
  std::vector<int> sortedSet = assignmentSet;
  std::sort(sortedSet.begin(), sortedSet.end());
  ssig::Cluster cluster;
  for (int i = 0; i < sortedSet.size() - 3; ++i) {
    cluster.push_back(sortedSet[i]);
    auto feat1 = features.row(sortedSet[i]);
    auto feat2 = features.row(sortedSet[i + 1]);
    auto feat3 = features.row(sortedSet[i + 2]);
    auto feat4 = features.row(sortedSet[i + 3]);

    cv::Mat_<float> out1, out2, out3;
    cv::matchTemplate(feat1, feat2, out1, cv::TM_SQDIFF_NORMED);
    cv::matchTemplate(feat1, feat3, out2, cv::TM_SQDIFF_NORMED);
    cv::matchTemplate(feat1, feat4, out3, cv::TM_SQDIFF_NORMED);
    if (out1[0][0] > 0.50) {
      initialClustering.push_back(cluster);
      cluster.clear();
    } else if (out2[0][0] >= 0.50) {
      cluster.push_back(sortedSet[++i]);
      initialClustering.push_back(cluster);
      cluster.clear();
    } else if (out3[0][0] >= 0.50) {
      cluster.push_back(sortedSet[++i]);
      cluster.push_back(sortedSet[++i]);
      initialClustering.push_back(cluster);
      cluster.clear();
    }
  }
}