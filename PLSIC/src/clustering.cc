#include "clustering.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <ml/svm_classifier.hpp>
#include <ml/pls_classifier.hpp>

#include <fstream>
#include <random>

#include "extraction.h"
#include "Initializator.h"


void Clustering::execute(ssig::PLSImageClustering& plsic,
                         const std::string& initializationMode,
                         const std::string& path,
                         const std::string& modelname){


  cv::Mat_<float> features;
  std::vector<std::string> sampleName;
  std::vector<cv::Rect> samplePatch;
  std::vector<float> sampleScales;
  setup(path,
        features,
        sampleName,
        samplePatch,
        sampleScales);

  auto discovery = split(features);
  plsic.setDiscoveryConfiguration(discovery);
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
  } else if(initializationMode == "oss"){
    Ossigiltering init;
    RandomInitializator rnd;
    auto rndInit = rnd(features, discovery[0]);
    initialClustering = init(features, discovery, rndInit);
  }

  cv::Mat vis;
  Clustering::visualizeAllClusters
    (initialClustering, sampleName, samplePatch, sampleScales, "begin");

  plsic.setInitialClustering(initialClustering);
  std::vector<ssig::Cluster> natVector = {{},{}};
  cv::Mat_<float> neg;
  plsic.addNaturalWorld(neg, natVector);


  cluster(features,
          sampleName,
          samplePatch,
          sampleScales,
          plsic
  );
  auto state = plsic.getClustering();
  visualizeAllClusters(state, sampleName, samplePatch, sampleScales, "plsic");

  plsic.save(modelname, "root");
}

void Clustering::execute(ssig::Singh& singh,
                         const std::string& path,
                         const std::string& pathNegatives,
                         const std::string& modelname){
  ssig::SVMClassifier svm;
  svm.setC(0.1f);
  svm.setKernelType(cv::ml::SVM::LINEAR);
  svm.setModelType(cv::ml::SVM::C_SVC);
  svm.setEpsilon(0.0001f);
  svm.setMaxIterations(static_cast<int>(1e5));
  svm.setTermType(cv::TermCriteria::COUNT);

  singh.setClassifier(svm);

  singh.setLambda(1.0);
  singh.setMValue(5);
  singh.setMaxIterations(8);

  cv::Mat_<float> features;
  std::vector<std::string> sampleName;
  std::vector<cv::Rect> samplePatch;
  std::vector<float> sampleScales;
  cv::Mat_<float> neg;
  setup(path,
        features,
        sampleName,
        samplePatch,
        sampleScales);

  setup(pathNegatives, neg);

  auto discovery = split(features);
  singh.setInitialK(features.rows / 4);
  singh.setDiscoveryConfiguration(discovery);
  std::vector<ssig::Cluster> initialClustering;
  SinghInitializator init;
  initialClustering = init(features, discovery[0]);


  cv::Mat vis;
  Clustering::visualizeAllClusters
    (initialClustering, sampleName, samplePatch, sampleScales, "begin");

  singh.setInitialClustering(initialClustering);

  auto natVector = split(neg);
  singh.addNaturalWorld(neg, natVector);


  cluster(features,
          sampleName,
          samplePatch,
          sampleScales,
          singh);

  auto state = singh.getClustering();
  visualizeAllClusters(state, sampleName, samplePatch, sampleScales, "singh");

  singh.save(modelname, "root");
}

void Clustering::execute(ssig::Kmeans& kmeans,
                         const std::string& path,
                         const std::string& modelname) const {
  cv::Mat_<float> features;
  std::vector<std::string> sampleName;
  std::vector<cv::Rect> samplePatch;
  std::vector<float> sampleScales;
  setup(path,
        features,
        sampleName,
        samplePatch,
        sampleScales);
  kmeans.setFlags(cv::KMEANS_RANDOM_CENTERS);
  cluster(features, kmeans);
  cv::Mat vis;
  Clustering::visualizeAllClusters
    (kmeans.getClustering(), sampleName, samplePatch, sampleScales, "kmeans");
  ssig::PLSClassifier pls;
  pls.setNumberOfFactors(5);
  ssig::OAAClassifier oaa(pls);
  cv::Mat_<int> labels;
  cv::Mat_<float> classifierInput;

  int labelId = -1;
  for(auto& cluster : kmeans.getClustering()){
    ++labelId;
    for(auto& id : cluster){
      classifierInput.push_back(features.row(id));
      labels.push_back(labelId);
    }
  }
  pls.save("kmeans_pls.yml", "root");

  kmeans.save(modelname, "root");
}

void Clustering::setup(const std::string& listFilename,
                       cv::Mat_<float>& inp,
                       std::vector<std::string>& sampleImagenames,
                       std::vector<cv::Rect>& samplePatches,
                       std::vector<float>& sampleScales) const{
  std::vector<cv::String> listing;
  std::ifstream listfile(listFilename);
  std::string filename = "";
  while(std::getline(listfile, filename)){
    cv::Mat_<float> features;
    std::string imagename;
    std::vector<cv::Rect> patches;
    std::vector<float> scales;
    Extraction::readFeatures(filename, features, imagename, patches, scales);
    inp.push_back(features);
    for(int i = 0; i < features.rows; ++i){
      samplePatches.push_back(patches[i]);
      sampleScales.push_back(scales[i]);
      sampleImagenames.push_back(imagename);
    }
  }
  int maxLen = MIN(inp.rows, 40000);
  cv::Mat_<int> shuffledOrdering(inp.rows, 1);
  std::vector<int> shuffledOrderingVector(inp.rows, 0);
  for(int i = 0; i < inp.rows; ++i){
    shuffledOrdering[i][0] = i;
  }
  cv::randShuffle(shuffledOrdering, 3.0);
  for(int i = 0; i < inp.rows; ++i){
    shuffledOrderingVector[i] = shuffledOrdering[i][0];
  }
  cv::Mat_<float> shuffledInp;
  ssig::Util::reorder(inp, shuffledOrdering, shuffledInp);
  std::vector<std::string> sampleImagenamesShuffled;
  ssig::Util::reorder(sampleImagenames, shuffledOrderingVector, sampleImagenamesShuffled);
  std::vector<cv::Rect> samplePatchesShuffled;
  ssig::Util::reorder(samplePatches, shuffledOrderingVector, samplePatchesShuffled);
  std::vector<float> sampleScalesShuffled;
  ssig::Util::reorder(sampleScales, shuffledOrderingVector, sampleScalesShuffled);
  sampleScales = sampleScalesShuffled;
  samplePatches = samplePatchesShuffled;
  sampleImagenames = sampleImagenamesShuffled;

  sampleScales.resize(maxLen);
  samplePatches.resize(maxLen);
  sampleImagenames.resize(maxLen);
  cv::resize(shuffledInp, inp, {shuffledInp.cols, maxLen});
}

void Clustering::setup(const std::string& pathToFeatures, cv::Mat_<float>& inp) const{
  std::vector<std::string> sampleImagenames;
  std::vector<cv::Rect> samplePatches;
  std::vector<float> sampleScales;
  setup(pathToFeatures, inp, sampleImagenames, samplePatches, sampleScales);
}

std::vector<std::vector<int>> Clustering::split(cv::Mat_<float>& inp){
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

void Clustering::cluster(cv::Mat_<float>& input, ssig::Clustering& clustering) {
  clustering.learn(input);
}

void Clustering::cluster(cv::Mat_<float>& input,
                         const std::vector<std::string>& mapSampleImagename,
                         const std::vector<cv::Rect>& mapSamplePatch,
                         std::vector<float>& sampleScales,
                         ssig::ClassifierClustering& clustering) {
  //int key = 32;
  cv::Mat vis;
  fprintf(stdout, "Method setup\n");
  clustering.setup(input);
  fprintf(stdout, "Setup Over\n");

  //int it = 0;
  do{
    auto state = clustering.getClustering();
    int maxSize = 0;
    for(auto& c : state)
      if(c.size() > maxSize)
        maxSize = static_cast<int>(c.size());

    /*visualizeAllClusters(state, mapSampleImagename, mapSamplePatch,
                         sampleScales, 80 * maxSize, vis);
    char number[10];
    sprintf(number, "%03d", ++it);
    cv::imwrite("Clustering" + std::string(number) + ".jpg", vis);*/
    //if(key != 113){//if key != q then visualize
    //  for(auto& cluster : state){
    //    if(key == 32){//key == space then iterate image
    //      visualizeCluster(cluster, mapSampleImagename, mapSamplePatch, vis);
    //      cv::imshow("cluster viewer", vis);
    //      key = cv::waitKey(0);
    //    } else if(key == 110){
    //      //key == n then it stops iterating over clusters and iterates the method
    //      break;
    //    } else{
    //      break;
    //    }
    //  }
    //}
    fprintf(stdout, "Iterating\n");
  } while(!clustering.iterate());

  //key = 32;
  //for(auto& cluster : state){
  //if(key == 32){//key == space then iterate image
  //  visualizeCluster(cluster, mapSampleImagename, mapSamplePatch, vis);
  //  cv::imshow("cluster viewer - End", vis);
  //  key = cv::waitKey(0);
  //} else if(key == 110){
  //  //key == n then it stops iterating over clusters and iterates the method
  //  break;
  //}
  //}
  //cv::destroyAllWindows();
}

void Clustering::visualizeAllClusters(const std::vector<ssig::Cluster>& clusters,
                                      const std::vector<std::string>& mapSampleImagename,
                                      const std::vector<cv::Rect>& mapSamplePatch,
                                      std::vector<float>& sampleScales,
                                      const int cols,
                                      cv::Mat& visualization){
  const int clusterslen = static_cast<int>(clusters.size());
  visualization.create(80 * clusterslen, cols, CV_8UC3);
  int c = 0;
  for(auto& cluster : clusters){
    const int len = static_cast<int>(cluster.size());
    cv::Mat vis;
    visualizeCluster(cluster, mapSampleImagename, mapSamplePatch, sampleScales, vis);
    auto clustervis = visualization(cv::Rect(0, c * 80, 80 * len, 80));
    vis.copyTo(clustervis);
    ++c;
  }

}

void Clustering::visualizeAllClusters(const std::vector<ssig::Cluster>& clusters,
                                      const std::vector<std::string>& mapSampleImagename,
                                      const std::vector<cv::Rect>& mapSamplePatch,
                                      std::vector<float>& sampleScales,
                                      const std::string& outdir){
  int it = 0;
  for(auto& cluster : clusters){
    cv::Mat vis;
    visualizeCluster(cluster, mapSampleImagename, mapSamplePatch, sampleScales, vis);
    char number[10];
    sprintf(number, "%05d", ++it);
    cv::imwrite(outdir + "\\cluster_" + number + ".jpg", vis);
  }
}

void Clustering::visualizeCluster(const std::vector<int>& cluster,
                                  const std::vector<std::string>& mapSampleImagename,
                                  const std::vector<cv::Rect>& mapSamplePatch,
                                  std::vector<float>& sampleScales,
                                  cv::Mat& visualization){
  int len = static_cast<int>(cluster.size());
  visualization.create(80, len * 80, CV_8UC3);

  int c = 0;
  for(auto id : cluster){
    auto name = mapSampleImagename[id];
    auto patch = mapSamplePatch[id];
    auto imgMat = cv::imread(name);
    cv::Size_<float> imgSize = {static_cast<float>(imgMat.cols),
        static_cast<float>(imgMat.rows)};
    cv::Size_<float> scaledSize = imgSize * sampleScales[id];
    cv::Mat scaledImage;
    cv::resize(imgMat, scaledImage, scaledSize, 0, 0, cv::INTER_CUBIC);
    auto patchImg = scaledImage(patch);
    cv::resize(patchImg, patchImg, {80, 80}, 0, 0, cv::INTER_CUBIC);
    auto patchVis = visualization(cv::Rect(80 * c, 0, 80, 80));
    patchImg.copyTo(patchVis);
    ++c;
  }
}
