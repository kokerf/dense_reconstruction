#ifndef DENSE_RECONSTRUCTION_FRAME_H
#define DENSE_RECONSTRUCTION_FRAME_H

#include <memory>

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <sophus/se3.h>

class Frame{
public:

    typedef std::shared_ptr<Frame> Ptr;

    Frame(cv::Mat& img, int max_level, cv::Mat& K, Sophus::SE3& pose);

    int createPyramid(const cv::Mat& img, std::vector<cv::Mat>& img_pyr, const uint16_t nlevels, const cv::Size min_size);

    void conv_32f(const cv::Mat& src, cv::Mat& dest, const cv::Mat& kernel, const int div);

    void makeBorders(const cv::Mat& src, cv::Mat& dest, const int col_side, const int row_side);

    inline long int id(){return id_;}

    inline Sophus::SE3 getPose() {return pose_;}

    inline const cv::Mat& getImageInLevel(int i) const { return img_pyr_[i];}

    inline const cv::Mat& getGradxInLevel(int i) const { return grad_x_pyr_[i];}

    inline const cv::Mat& getGradyInLevel(int i) const { return grad_y_pyr_[i];}

    inline Eigen::Matrix3d camK(){return eigK_;}

    inline Eigen::Vector3d c2w(const Eigen::Vector3d v){return T_w_c_*v;}

    inline Eigen::Vector3d lift(const int x, const int y){return Eigen::Vector3d(ifx_*x+icx_,ify_*y+icy_,1);}

    inline Eigen::Vector2d project(const Eigen::Vector3d p){return Eigen::Vector2d(fx_*p[0]/p[2]+cx_,fy_*p[1]/p[2]+cy_);}

    inline bool isInFrame(const double x, const double y, const int boundary){
        if(x >= boundary && x < width_-boundary && y >= boundary && y < height_-boundary)
            return true;
        return false;
    }

private:
    static long int frame_counter;

    const long int id_;

    cv::Mat K_;
    cv::Mat invK_;

    Eigen::Matrix3d eigK_;

    double fx_, fy_, cx_, cy_;
    double ifx_, ify_, icx_, icy_;

    int width_;
    int height_;

    std::vector<cv::Mat> img_pyr_;
    std::vector<cv::Mat> grad_x_pyr_;
    std::vector<cv::Mat> grad_y_pyr_;
    int level_;
    Sophus::SE3 pose_;
    Sophus::SE3 T_w_c_;
};


#endif //DENSE_RECONSTRUCTION_FRAME_H
