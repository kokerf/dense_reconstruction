#include <stdlib.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "depth_filter.h"
#include "config.h"

long int Seed::seed_counter = 0;

//!
//! Seed
//!
Seed::Seed(Frame::Ptr _frame, Eigen::Vector3d _f, double _depth_mean, double _sigma2):
        frame(_frame),
        id(seed_counter++),
        f(_f),
        mu(1.0/_depth_mean),
        sigma2(_sigma2)
{}


//!
//! DepthFilter
//!
DepthFilter::DepthFilter(int min_level, int max_level):
    min_level_(min_level), max_level_(max_level), width_(Config::imageWidth()), height_(Config::imageHeight())
{}

void DepthFilter::initSeeds(Frame::Ptr frame)
{
    frame_sequence_.clear();
    frame_sequence_.push_back(frame);

    //const cv::Mat img = frame->getImageInLevel(0);
    const cv::Mat gradx = frame->getGradxInLevel(0);
    const cv::Mat grady = frame->getGradyInLevel(0);

    const int min_grad2 = Config::seedMinGrad()*Config::seedMinGrad();

    const int cols = Config::imageWidth() - 1;
    const int rows = Config::imageHeight() - 1;

    for (int r = 1; r < rows; ++r)
    {
        const float* gradx_ptr = gradx.ptr<float>(r);
        const float* grady_ptr = grady.ptr<float>(r);

        for (int c = 1; c < cols; ++c)
        {
            const float dx = gradx_ptr[c];
            const float dy = grady_ptr[c];

            const float grad2 = dx*dx + dy*dy;

            if(grad2 > min_grad2)
            {
                Eigen::Vector3d ftr = frame->lift(c, r);
                float depth = 0.5f + 1.0f * ((rand() % 100001) / 100000.0f);
                seeds_.push_back(Seed(frame, ftr, depth, Config::seedInitVar2()));
            }

        }
    }
}

void DepthFilter::addFrame(Frame::Ptr new_frame)
{
    if(frame_sequence_.empty())
    {
        initSeeds(new_frame);
        return;
    }

    std::vector<Seed>::iterator seed_itr = seeds_.begin();
    std::vector<Seed>::iterator seed_end = seeds_.end();
    for(; seed_itr != seed_end; ++seed_itr)
    {
        //! transform from ref to cur
        const Sophus::SE3 T_cur_ref =  new_frame->getPose() * seed_itr->frame->getPose().inverse();

        Eigen::Vector3d ft_ref = seed_itr->f;

        Eigen::Vector3d p3d_ref =  ft_ref / seed_itr->mu;

        Eigen::Vector3d p3d_cur = T_cur_ref*p3d_ref;

        if(p3d_cur[2] <= 0.0)
            continue;

        Eigen::Vector2d px_cur = new_frame->project(p3d_cur) + Eigen::Vector2d(0.5,0.5);
        if(!new_frame->isInFrame(Eigen::Vector2i(px_cur[0], px_cur[1]), border_))
            continue;

        double idepth_max = seed_itr->mu+3*sqrtf(seed_itr->sigma2);
        double idepth_min = std::max(seed_itr->mu-3*sqrtf(seed_itr->sigma2), 0.000001);

        Eigen::Vector3d ft_cur;
        double ncc_err = 0;
        bool succeed = searchEpipolarLine(seed_itr->frame, new_frame, T_cur_ref, ft_ref,
                                          1.0/idepth_max, 1.0/idepth_min,
                                          ft_cur, ncc_err);

        if(!succeed)
            continue;

        //! transform from cur to ref
        const Sophus::SE3 T_ref_cur = seed_itr->frame->getPose() * new_frame->getPose().inverse();
        double z = triangulate(ft_ref, ft_cur, T_ref_cur);

        double var = calcVariance(T_cur_ref.translation(), ft_ref, z, Config::pixelError());

        double var_inverse = 0.5 * (1.0/std::max(0.0000001, z-var) - 1.0/(z+var));
        seed_itr->update(z, var_inverse*var_inverse);
    }
}

bool DepthFilter::searchEpipolarLine(const Frame::Ptr reference,
                                     const Frame::Ptr current,
                                     const Sophus::SE3 T_cur_ref,
                                     const Eigen::Vector3d &ft_ref,
                                     const double depth_min,
                                     const double depth_max,
                                     Eigen::Vector3d &ft_cur,
                                     double &ncc_err)
{

    //! in ref
    Eigen::Vector2d px = reference->project(ft_ref);
    Eigen::Vector2i px_i(px[0]+0.5, px[1]+0.5);
    //! in cur
    Eigen::Vector2d px_near = current->project(T_cur_ref * (depth_min * ft_ref));
    Eigen::Vector2d px_far = current->project(T_cur_ref * (depth_max * ft_ref));

    Eigen::Vector2d epline = px_far - px_near;

    //! check for angle
    const cv::Mat gradx_cur = current->getGradxInLevel(0);
    const cv::Mat grady_cur = current->getGradyInLevel(0);

    const double dx = interpolateMat_32f(gradx_cur,px[0], px_i[1]);
    const double dy = interpolateMat_32f(grady_cur,px[0], px_i[1]);

    double dot2 = dx*epline[0] + dy*epline[1];
    dot2 = dot2 * dot2;
    const double epline2 = epline[0]*epline[0] + epline[1]*epline[1];
    const double grad_prj2 = dot2/epline2;
    //! costh*|g| = g * pl / |pl|
    if(grad_prj2 < Config::minEplGrad2())
        return false;

    const double grad2 = dx*dx + dy*dy;
    const double costhta2 = grad_prj2/grad2;
    //! costh = g * pl / (|pl|*|g|)
    if(costhta2 < Config::minEplAngle2())
        return false;

    epline.normalize();

    rangePoint(px_near, epline);
    rangePoint(px_far, epline);

    Eigen::Vector2i px_near_i(px_near[0]+0.5, px_near[1]+0.5);
    Eigen::Vector2i px_far_i(px_far[0]+0.5, px_far[1]+0.5);
    if(!current->isInFrame(px_near_i, border_) || !current->isInFrame(px_far_i, border_))
        return false;

    Eigen::Vector2d new_epline = px_far - px_near;
    double length_epl = std::sqrt(new_epline[0]*new_epline[0] + new_epline[1]*new_epline[1]);
    if(length_epl < 2.0)
        return false;//TODO

    const cv::Mat img_ref = reference->getImageInLevel(0);
    const cv::Mat img_cur = current->getImageInLevel(0);

    int end_epl = static_cast<int>(length_epl + 1.0);
    double min_err = std::numeric_limits<double>::max();
    int best_match_i = 0;
    for(int i = 0; i < end_epl; ++i)
    {
        Eigen::Vector2i px_cur(px_near_i[0] + i * epline[0], px_near_i[1] + i * epline[1]);
        double err = NCC(img_ref, img_cur, px_i, px_cur);

        if(err < min_err)
        {
            min_err = err;
            best_match_i = i;
        }
    }

    if(min_err > Config::maxNCCError())
        return false;

    // TODO
    ft_cur = current->lift(px_near_i[0] + best_match_i*epline[0], px_near_i[1] + best_match_i*epline[1]);

    //showEpipolarLine(img_ref, img_cur, px, px_near, px_far);

    return true;
}

double DepthFilter::triangulate(const Eigen::Vector3d &ft_ref,
                                const Eigen::Vector3d &ft_cur,
                                const Sophus::SE3 &T_ref_cur)
{
    //! d_ref* ft_ref = d_cur* R * ft_cur + t
    //! => [ft_ref -R*ft_cur] [d_ref d_cur]^T = t
    //！ A x = b
    Eigen::Vector3d t = T_ref_cur.translation();
    Eigen::Matrix3d R = T_ref_cur.rotation_matrix();
    Eigen::Vector3d R_ft_cur = R * ft_cur;

    std::cout << "ref: " << ft_ref.transpose()  << std::endl;
    std::cout << "cur: " << ft_cur.transpose()  << std::endl;
    std::cout << "t: " << t.transpose()  << std::endl;
    std::cout << "R_ft_cur: " << R_ft_cur.transpose()  << std::endl;

    double b[2] = {t[0], t[1]};
    double A[4] = {ft_ref[0], -R_ft_cur[0], ft_ref[1], -R_ft_cur[1]};
    double det = A[0]*A[3] - A[2]*A[1];
    double d_ref = (b[0]*A[3] - b[1]*A[1])/det;
    double d_cur = (A[0]*b[1] - A[2]*b[0])/det;

    Eigen::Vector3d ft1 = d_ref * ft_ref;
    Eigen::Vector3d ft2 = d_cur * R * ft_cur + t;
    Eigen::Vector3d estimate_ft = 0.5 * (ft1 + ft2);

    double z = estimate_ft.norm();

    return z;
}

void DepthFilter::rangePoint(Eigen::Vector2d& px, const Eigen::Vector2d& dir)
{
    float delta = border_ - px[0];
    if(delta > 0.0)
    {
        px += delta/dir[0] * dir;
    }

    delta = border_ - px[1];
    if(delta > 0.0)
    {
        px += delta/dir[1] * dir;
    }

    delta = (width_- 1- border_) - px[0];
    if(delta < 0.0)
    {
        px += delta/dir[0] * dir;
    }

    delta = (width_- 1- border_) - px[1];
    if(delta < 0.0)
    {
        px += delta/dir[1] * dir;
    }
}

double DepthFilter::NCC(const cv::Mat &img_ref,
                        const cv::Mat &img_cur,
                        const Eigen::Vector2i &pt_ref,
                        const Eigen::Vector2i &pt_cur)
{

    const double ncc_area = win_size_*win_size_;
    std::vector<double> value_ref(ncc_area);
    std::vector<double> value_cur(ncc_area);
    double means_ref = 0.0;
    double means_cur = 0.0;
    int k = 0;
    for(int y = 0; y < win_size_; ++y)
    {
        const float* ref_ptr = img_ref.ptr<float>(pt_ref[1]+y);
        const float* cur_ptr = img_cur.ptr<float>(pt_cur[1]+y);
        for(int x = 0; x < win_size_; ++x)
        {
            value_ref[k] = *(ref_ptr+x);
            value_cur[k] = *(cur_ptr+x);
            means_ref+=value_ref[k];
            means_cur+=value_cur[k];
            k++;
        }
    }

    means_ref /= ncc_area;
    means_cur /= ncc_area;

    double AB = 0.0;
    double AA = 0.0;
    double BB = 0.0;
    for(int i = 0; i < ncc_area; ++i)
    {
        const double val0 = value_ref[i] - means_ref;
        const double val1 = value_cur[i] - means_cur;

        AB += val0*val1;
        AA += val0*val0;
        BB += val1*val1;
    }

    return AB/std::sqrt(AA*BB + std::numeric_limits<double>::epsilon());
}

double DepthFilter::calcVariance(const Eigen::Vector3d &f,
                                 const Eigen::Vector3d &t,
                                 const double z,
                                 double px_error)
{
    const Eigen::Vector3d a = z * f - t;
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos(f.dot(t)/t_norm); // dot product
    double beta = acos(a.dot(-t)/(t_norm*a_norm)); // dot product
    double beta_plus = beta + atan(px_error/(2.0*480))*2.0;
    double gamma_plus = M_PI-alpha-beta_plus; // triangle angles sum to PI
    double z_plus = t_norm*sin(beta_plus)/sin(gamma_plus); // law of sines
    return (z_plus - z); // tau
}

DepthFilter DepthFilter::getWarpMatrixAffine(
        const Frame::Ptr reference, const Frame::Ptr current,
        const Eigen::Vector2d& px_ref, const Eigen::Vector3d& f_ref,
        const double depth_ref, Eigen::Matrix2d& A_cur_ref)
{
    // Compute affine warp matrix A_ref_cur
    const int halfpatch_size = 5;
    const Eigen::Vector3d xyz_ref = depth_ref*f_ref;
    Eigen::Vector3d xyz_du_ref = reference->lift(px_ref[0]+halfpatch_size, px_ref[1]);
    Eigen::Vector3d xyz_dv_ref = reference->lift(px_ref[0], px_ref[1]+halfpatch_size);
    xyz_du_ref *= xyz_ref[2]/xyz_du_ref[2];
    xyz_dv_ref *= xyz_ref[2]/xyz_du_ref[2];

    const Sophus::SE3 T_cur_ref = current->getPose() * reference->getPose().inverse();
    const Eigen::Vector2d px_cur(current->project(T_cur_ref*(xyz_ref)));
    const Eigen::Vector2d px_du(current->project(T_cur_ref*(xyz_du_ref)));
    const Eigen::Vector2d px_dv(current->project(T_cur_ref*(xyz_dv_ref)));
    A_cur_ref.col(0) = (px_du - px_cur)/halfpatch_size;
    A_cur_ref.col(1) = (px_dv - px_cur)/halfpatch_size;
}

void DepthFilter::getMapPoints(std::vector<Eigen::Vector3d> &points)
{
    points.resize(mappoints_.size());
    for (int i = 0; i < mappoints_.size(); ++i)
    {
        points[i] = mappoints_[i];
    }
}

void DepthFilter::getAllPoints(std::vector<Eigen::Vector3d> &points, std::vector<double> &sigma2)
{
    points.resize(seeds_.size()+mappoints_.size());
    sigma2.resize(seeds_.size()+mappoints_.size(), 0.0);
    for (int i = 0; i < seeds_.size(); ++i)
    {
        points[i] = seeds_[i].toMapPoint();
        sigma2[i] = seeds_[i].sigma2;
    }

    for (int j = 0; j < mappoints_.size(); ++j)
    {
        points[j] = mappoints_[j];
    }
}

//! https://github.com/uzh-rpg/rpg_vikit/blob/master/vikit_common/include/vikit/vision.h
//! WARNING This function does not check whether the x/y is within the border
inline float DepthFilter::interpolateMat_32f(const cv::Mat& mat, const float u, const float v)
{
    assert(mat.type() == CV_32F);
    float x = floor(u);
    float y = floor(v);
    float subpix_x = u - x;
    float subpix_y = v - y;
    float wx0 = 1.0 - subpix_x;
    float wx1 = subpix_x;
    float wy0 = 1.0 - subpix_y;
    float wy1 = subpix_y;

    float val00 = mat.at<float>(y, x);
    float val10 = mat.at<float>(y, x + 1);
    float val01 = mat.at<float>(y + 1, x);
    float val11 = mat.at<float>(y + 1, x + 1);
    return (wx0*wy0)*val00 + (wx1*wy0)*val10 + (wx0*wy1)*val01 + (wx1*wy1)*val11;
}

void DepthFilter::showEpipolarMatch(const cv::Mat &ref,
                                    const cv::Mat &curr,
                                    const Eigen::Vector2d &px_ref,
                                    const Eigen::Vector2d &px_curr)
{
    cv::Mat ref_show, curr_show;
    cv::cvtColor( ref, ref_show, CV_GRAY2BGR );
    cv::cvtColor( curr, curr_show, CV_GRAY2BGR );

    cv::circle( ref_show, cv::Point2f(px_ref(0,0), px_ref(1,0)), 5, cv::Scalar(0,0,250), 2);
    cv::circle( curr_show, cv::Point2f(px_curr(0,0), px_curr(1,0)), 5, cv::Scalar(0,0,250), 2);

    cv::imshow("ref", ref_show );
    cv::imshow("curr", curr_show );
    cv::waitKey(1);
}

void DepthFilter::showEpipolarLine(const cv::Mat &ref,
                                   const cv::Mat &curr,
                                   const Eigen::Vector2d &px_ref,
                                   const Eigen::Vector2d &px_min_curr,
                                   const Eigen::Vector2d &px_max_curr)
{

    cv::Mat ref_show, curr_show;
    cv::cvtColor( ref, ref_show, CV_GRAY2BGR );
    cv::cvtColor( curr, curr_show, CV_GRAY2BGR );

    cv::circle( ref_show, cv::Point2f(px_ref(0,0), px_ref(1,0)), 5, cv::Scalar(0,255,0), 2);
    cv::circle( curr_show, cv::Point2f(px_min_curr(0,0), px_min_curr(1,0)), 5, cv::Scalar(0,255,0), 2);
    cv::circle( curr_show, cv::Point2f(px_max_curr(0,0), px_max_curr(1,0)), 5, cv::Scalar(0,255,0), 2);
    cv::line( curr_show, cv::Point2f(px_min_curr(0,0), px_min_curr(1,0)), cv::Point2f(px_max_curr(0,0), px_max_curr(1,0)), cv::Scalar(0,255,0), 1);

    cv::imshow("ref", ref_show );
    cv::imshow("curr", curr_show );
    cv::waitKey(1);
}