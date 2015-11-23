#include "extraction.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <core/sampling.hpp>
#include <descriptors/hog_features.hpp>


void Extraction::extractFeatures(const std::string& structureFile,
                                 const std::string& outDir,
                                 const cv::Size& windowSize,
                                 const std::pair<float, float>& stride){
  cv::FileStorage datasetStructure;
  datasetStructure.open(structureFile, cv::FileStorage::READ);
  auto mainNode = datasetStructure["files"];
  int nImages = 0, nPatches = 0;

  std::vector<float> scales = {0.25f, 0.4f, 0.55f, 0.7f, 0.85f, 1.0f, 1.25f};
  for(auto it = mainNode.begin(); it != mainNode.end(); ++it){
    assert(!(*it).isNone());

    std::string filename = std::string(*it);
    cv::Mat imgMat = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
    if(imgMat.empty())
      exit(EXIT_FAILURE);
    fprintf(stdout, "Processing [%s]\n", filename.c_str());
    std::vector<cv::Rect> patches;

    std::vector<cv::Mat_<float>> descriptors;
    std::vector<float> scalesVec;
    for(auto& scale : scales){
      cv::Mat scaledImg;
      cv::Size_<float> imgSize = {static_cast<float>(imgMat.cols), static_cast<float>(imgMat.rows)};
      auto scaledSize = imgSize * scale;
      cv::resize(imgMat, scaledImg, scaledSize, 0, 0, cv::INTER_CUBIC);
      std::vector<cv::Rect> scalePatches;
      try{
        scalePatches = ssig::Sampling::sampleImage(scaledImg, 8, { 80, 80 });
      } catch(std::exception e){
        //no patch for this scale
      }

      ssig::HOG hog(scaledImg);

      for(auto& patch : scalePatches){
        cv::Mat out;
        cv::Size blockSz(patch.width / 2, patch.height / 2);

        hog.setBlockConfiguration(blockSz);
        hog.setBlockStride(blockSz);
        hog.setCellConfiguration({ 4, 4 });
        hog.setClipping(0.2f);
        hog.setNumberOfBins(9);

        hog.extract({patch}, out);
        float energyThreshold = (out.cols * 0.01f);
        if(cv::norm(out, cv::NORM_L1) >= energyThreshold){
          descriptors.push_back(out);
          patches.push_back(patch);
          scalesVec.push_back(scale);
        }
      }
    }
    writeDescriptor(descriptors, patches, scalesVec, outDir, filename);
    const int len = static_cast<int>(patches.size());
    float avgPatches = nPatches / static_cast<float>(nImages);
    fprintf(stdout, "[%d] patches avg[%f]\n", len, avgPatches);
    nPatches += len;
    patches.clear();
    ++nImages;
  }
  float avgPatches = nPatches / static_cast<float>(nImages);
  fprintf(stdout, "Average number of patches per image is %f\n", avgPatches);
}

void Extraction::writeDescriptor(
  const std::vector<cv::Mat_<float>>& descriptors,
  const std::vector<cv::Rect>& rects,
  const std::vector<float>& scales,
  const std::string& outDir, std::string& outputname){

  std::string imgName = outputname.substr(outputname.find_last_of("\\") + 1);
  imgName = imgName.substr(0, imgName.find_last_of("."));
  std::string outName = outDir + imgName + ".yml";


  cv::FileStorage stg;
  stg.open(outName, cv::FileStorage::WRITE);
  stg << "imageName" << outputname;
  stg << "features" << "{";
  for(size_t i = 0; i < descriptors.size(); ++i){
    int pos = static_cast<int>(i);
    cv::Mat m = descriptors[pos];
    stg << "patch" + std::to_string(pos) << "{";
    stg << "feature" << m;
    stg << "rect" << rects[pos];
    stg << "scale" << scales[pos];
    stg << "}";
  }
  stg << "}";
  stg.release();
}

void Extraction::readFeatures(std::string& featurefilename,
                              cv::Mat_<float>& features,
                              std::string& imagename,
                              std::vector<cv::Rect>& patches,
                              std::vector<float>& scales){
  fprintf(stdout, "Reading File [%s]\n", featurefilename.c_str());
  cv::FileStorage featfile(featurefilename, cv::FileStorage::READ);
  auto featNode = featfile["features"];
  float scale;
  for(auto it = featNode.begin(); it != featNode.end(); ++it){
    cv::Mat feat;
    cv::Rect patch;
    auto node = *it;
    node["feature"] >> feat;
    feat = feat / (cv::norm(feat) + FLT_EPSILON);
    node["rect"] >> patch;
    node["scale"] >> scale;

    featfile["imageName"] >> imagename;
    scales.push_back(scale);
    features.push_back(feat);
    patches.push_back(patch);
  }
}
