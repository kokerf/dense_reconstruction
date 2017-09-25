#ifndef DENSE_RECONSTRUCTION_DEPTH_FILTER_H
#define DENSE_RECONSTRUCTION_DEPTH_FILTER_H

#include <opencv2/core.hpp>

#include <Eigen/Core>

#include <sophus/se3.h>

#include "frame.h"

class Seed{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Seed(Frame::Ptr _frame, Eigen::Vector3d _f, double _depth_mean, double _depth_min);

    inline void update(const double z, const double var2) {
        mu = (sigma2*mu + var2/z)/(sigma2 + var2);
        sigma2 = (sigma2 * var2)/(sigma2 + var2);
    }

    inline Eigen::Vector3d toMapPoint(){
        return frame->c2w(f) / mu;
    }

public:
    static long int seed_counter;

    Frame::Ptr frame;
    const long int id;

    double mu;
    double sigma2;

    double d_min;
    double d_max;
    Eigen::Vector3d f;
};


class DepthFilter{
public:
    DepthFilter(int min_level = 0, int max_level = 0);

    void addFrame(Frame::Ptr new_frame);

    void getMapPoints(std::vector<Eigen::Vector3d>& points);

    void getAllPoints(std::vector<Eigen::Vector3d>& points, std::vector<double>& sigma2);

    void showEpipolarMatch(const cv::Mat& ref, const cv::Mat& curr, const Eigen::Vector2d& px_ref, const Eigen::Vector2d& px_curr);

    void showEpipolarLine(const cv::Mat& ref, const cv::Mat& curr, const Eigen::Vector2d& px_ref, const Eigen::Vector2d& px_min_curr, const Eigen::Vector2d& px_max_curr);

 private:

    void initSeeds(Frame::Ptr frame);

    bool searchEpipolarLine(const Frame::Ptr reference, const Frame::Ptr current,
                            const Sophus::SE3 T_cur_ref, const Eigen::Vector3d &ft_ref,
                            const double depth_min, const double depth_max,
                            Eigen::Vector3d &ft_cur, double& ncc_err);

    void rangePoint(Eigen::Vector2d& px, const Eigen::Vector2d& dir);

    double NCC(const cv::Mat &img_ref, const cv::Mat &img_cur, const Eigen::Vector2i &pt_ref, const Eigen::Vector2i &pt_cur);

    float interpolateMat_32f(const cv::Mat& mat, const float u, const float v);

    double triangulate(const Eigen::Vector3d& ft_ref, const Eigen::Vector3d& ft_cur, const Sophus::SE3& T_ref_cur);

    double calcVariance(const Eigen::Vector3d& f, const Eigen::Vector3d& t, const double z, double px_error);

    DepthFilter getWarpMatrixAffine(
            const Frame::Ptr reference, const Frame::Ptr current,
            const Eigen::Vector2d& px_ref, const Eigen::Vector3d& f_ref,
            const double depth_ref,
            Eigen::Matrix2d& A_cur_ref);

private:
    int min_level_;
    int max_level_;

    const int border_ = 2;
    const int win_size_ = 5;
    const int width_;
    const int height_;


    cv::Mat reference_;

    std::vector<Frame::Ptr> frame_sequence_;

    std::vector<Seed> seeds_;

    std::vector<Eigen::Vector3d> mappoints_;

};


#endif //DENSE_RECONSTRUCTION_DEPTH_FILTER_H
