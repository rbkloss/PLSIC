#include <opencv2/core.hpp>
#include<fstream>
#include <set>
#include <unordered_map>


int main(int argc, char** argv){
  if(argc < 6){
    printf("Usage: exe path mask attributesFile classesFile outFile");
    return 0;
  }
  std::vector<cv::String> listing;
  std::string path = argv[1];
  std::string mask = argv[2];
  cv::glob(path + mask, listing);
  cv::FileStorage attributesFile(argv[3], cv::FileStorage::READ);
  std::string prefix = "voc";
  std::ifstream classesFile(argv[4]);
  // = { "horse", "motorbike", "bus", "train", "sofa", "diningtable" }
  std::set<std::string> trainLabels;
  while(classesFile){
    std::string label;
    classesFile >> label;
    trainLabels.insert(label);
  }
  int nImages = 0;
  std::unordered_map<std::string, int> labelStatistics;
  std::ofstream trainFiles(argv[5]);
  for(auto& filename : listing){
    auto root = filename.substr(0, filename.size() - 4);
    root = root.substr(root.find_last_of("\\") + 1);
    auto filenode = attributesFile[prefix + root];

    auto classesNode = filenode["classes"];
    std::vector<std::string> labels;
    bool found = false;
    for(auto i = classesNode.begin(); i != classesNode.end(); ++i){
      std::string label;
      *i >> label;
      labels.push_back(label);
      if(!found && trainLabels.find(label) != trainLabels.end()){
        trainFiles << filename << std::endl;
        found = true;
        nImages++;
      }
    }
    std::sort(labels.begin(), labels.end());
    auto it = std::unique(labels.begin(), labels.end());
    labels.resize(std::distance(labels.begin(), it));
    if (found) {
      for (auto& label : labels) {
        labelStatistics[label] = labelStatistics[label] + 1;
      }
    }
  }

  for(auto& stat : labelStatistics){
    printf("[%s] [%d]\n", stat.first.c_str(), stat.second);
  }
  printf("total:[%d]\n", nImages);

  trainFiles.close();
}
