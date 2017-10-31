#ifndef DENSE_RECONSTRUCTION_VIEWER_H
#define DENSE_RECONSTRUCTION_VIEWER_H

#include <vector>
#include <thread>
#include <mutex>
#include <Eigen/Core>
#include <opencv2/viz.hpp>

class Viewer {
public:
    Viewer(const std::string &window_name);

    void run();

    void updateMapPoint(std::vector<Eigen::Vector3d> &points, const std::vector<double> &vars);

private:
    std::vector<cv::Mat> clouds_;
    std::vector<cv::viz::Color> colors_;
    cv::viz::Viz3d window_;
    std::mutex mMutexCloud;
};

#endif //DENSE_RECONSTRUCTION_VIEWER_H
