#include <opencv2/imgproc.hpp>
#include "frame.h"

long int Frame::frame_counter = 0;

const cv::Mat kernel1 = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
const cv::Mat kernel2 = (cv::Mat_<float>(3,3)<< -1,-2,-1, 0, 0, 0, 1, 2, 1);

Frame::Frame(cv::Mat &img, int max_level, cv::Mat& K, Sophus::SE3& pose):
        id_(frame_counter++), pose_(pose), T_w_c_(pose_.inverse())
{
    K_ = K.clone();
    invK_ = K_.inv();
    fx_ = K_.at<double>(0,0);
    fy_ = K_.at<double>(1,1);
    cx_ = K_.at<double>(0,2);
    cy_ = K_.at<double>(1,2);

    ifx_ = invK_.at<double>(0,0);
    ify_ = invK_.at<double>(1,1);
    icx_ = invK_.at<double>(0,2);
    icy_ = invK_.at<double>(1,2);

    eigK_ << fx_,0,cx_,0,fy_,cy_,0,0,1;

    width_ = img.cols;
    height_ = img.rows;


    level_ = createPyramid(img, img_pyr_, max_level, cv::Size(40,40));

    grad_x_pyr_.resize(level_+1);
    grad_y_pyr_.resize(level_+1);
    for (int i = 0; i <= level_; ++i)
    {
        conv_32f(img_pyr_[i], grad_x_pyr_[i], kernel1, 8);
        conv_32f(img_pyr_[i], grad_y_pyr_[i], kernel2, 8);
    }
}

int Frame::createPyramid(const cv::Mat& img, std::vector<cv::Mat>& img_pyr, const uint16_t nlevels, const cv::Size min_size)
{
    assert(!img.empty());

    img_pyr.resize(nlevels+1);

    if(img.type() != CV_8UC1)
        cv::cvtColor(img, img_pyr[0], cv::COLOR_RGB2GRAY);
    else
        img.copyTo(img_pyr[0]);

    for(int i = 1; i <= nlevels; ++i)
    {
        cv::Size size(round(img_pyr[i - 1].cols >> 1), round(img_pyr[i - 1].rows >> 1));

        if(size.height < min_size.height || size.width < min_size.width)
        {
            img_pyr.resize(i);
            return i-1;
        }

        cv::resize(img_pyr[i - 1], img_pyr[i], size, 0, 0, cv::INTER_LINEAR);
    }

    return nlevels;
}

void Frame::conv_32f(const cv::Mat& src, cv::Mat& dest, const cv::Mat& kernel, const int div)
{
    assert(!src.empty());
    assert(src.type() == CV_8UC1);
    assert(kernel.type() == CV_32FC1);
    assert(kernel.cols == 3 && kernel.rows == 3);

    //! copy borders for image
    cv::Mat img_extend;
    makeBorders(src, img_extend, 1, 1);

    //! calculate the dest image
    float *kernel_data = (float*) kernel.data;

    dest = cv::Mat::zeros(src.size(), CV_32FC1);
    int u,v;
    const uint16_t src_cols = src.cols;
    const uint16_t src_rows = src.rows;
    for(int ir = 0; ir < src_rows; ++ir)
    {
        v = ir + 1;
        float* dest_ptr = dest.ptr<float>(ir);
        uint8_t* extd_ptr = img_extend.ptr<uint8_t>(v);
        for(int ic = 0; ic < src_cols; ++ic)
        {
            u = ic + 1;

            dest_ptr[ic] = kernel_data[0] * extd_ptr[u - 1 - src_cols]
                           + kernel_data[1] * extd_ptr[u - src_cols]
                           + kernel_data[2] * extd_ptr[u + 1 - src_cols]
                           + kernel_data[3] * extd_ptr[u - 1]
                           + kernel_data[4] * extd_ptr[u]
                           + kernel_data[5] * extd_ptr[u + 1]
                           + kernel_data[6] * extd_ptr[u - 1 + src_cols]
                           + kernel_data[7] * extd_ptr[u + src_cols]
                           + kernel_data[8] * extd_ptr[u + 1 + src_cols];

            dest_ptr[ic] /= div;
        }
    }
}

void Frame::makeBorders(const cv::Mat& src, cv::Mat& dest, const int col_side, const int row_side)
{
    assert(!src.empty());
    assert(col_side > 0 && row_side > 0);

    const uint16_t src_cols = src.cols;
    const uint16_t src_rows = src.rows;

    cv::Mat border = cv::Mat::zeros(cv::Size(src_cols + row_side *2, src_rows + col_side*2), CV_8UC1);
    src.copyTo(border.rowRange(col_side, col_side + src_rows).colRange(row_side, row_side + src_cols));
    for(int ir = 0; ir < col_side; ++ir)
    {
        src.row(0).copyTo(border.row(ir).colRange(row_side, row_side + src_cols));
        int ir_inv = border.rows - ir - 1;
        src.row(src_rows - 1).copyTo(border.row(ir_inv).colRange(row_side, row_side + src_cols));
    }

    for(int ic = 0; ic < row_side; ++ic)
    {
        border.col(row_side).copyTo(border.col(ic));
        int ic_inv = border.cols - ic - 1;
        border.col(row_side + src_cols - 1).copyTo(border.col(ic_inv));
    }

    dest = border.clone();
}
