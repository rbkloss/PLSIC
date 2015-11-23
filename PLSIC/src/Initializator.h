#ifndef _INITIALIZER_H_
#define _INITIALIZER_H_

#include <vector>
#include <ml/clustering.hpp>

class ClusteringInitializator{
  public:
  virtual ~ClusteringInitializator() = default;

  virtual std::vector<ssig::Cluster> operator()(const cv::Mat_<float>& inp,
    const std::vector<int>& assignmentSet)const = 0;
};

class RandomInitializator : public ClusteringInitializator {

public:
virtual std::vector<ssig::Cluster> operator()(const cv::Mat_<float>& inp, const std::vector<int>& assignmentSet) const override;
};

class KmeansInitializator : public ClusteringInitializator {
public:
virtual std::vector<ssig::Cluster> operator()(const cv::Mat_<float>& inp, const std::vector<int>& assignmentSet) const override;
};

class SinghInitializator : public ClusteringInitializator {
  public:
  virtual std::vector<ssig::Cluster> operator()(const cv::Mat_<float>& inp, const std::vector<int>& assignmentSet) const override;
};

class SplitInitializator : public ClusteringInitializator {

public:
virtual std::vector<ssig::Cluster> operator()(const cv::Mat_<float>& inp, const std::vector<int>& assignmentSet) const override;
};

class SplitJoinInitializator : public ClusteringInitializator {

  public:
  virtual std::vector<ssig::Cluster> operator()(const cv::Mat_<float>& inp, const std::vector<int>& assignmentSet) const override;
};

class Ossigiltering{
  public:
  std::vector<ssig::Cluster> operator()(const cv::Mat_<float>& inp,
    const std::vector<std::vector<int>>& discovery,
    const std::vector<ssig::Cluster>&initial);
};

class AvilaInitializator{
  public: 
  void operator()(const cv::Mat_<float>& features,
    const std::vector<int>& assignmentSet,
    std::vector<ssig::Cluster>& initialClustering)const;
};

#endif
