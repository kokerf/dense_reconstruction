#include "viewer.h"

Viewer::Viewer(const std::string &window_name):
    window_(window_name)
{
    window_.showWidget("Coordinate Frame", cv::viz::WCoordinateSystem(0.1));
}

void Viewer::run()
{

    while(!window_.wasStopped())
    {
        {
            std::unique_lock<std::mutex> lock(mMutexCloud);
            for(size_t i = 0; i < clouds_.size(); ++i)
            {
                cv::viz::WCloud cloud_widget(clouds_[i], colors_[i]);
                const std::string widget_name = "MapPoint" + std::to_string(i);
                window_.showWidget(widget_name, cloud_widget);
            }
        }

        window_.spinOnce(10, true);
    }
}

void Viewer::updateMapPoint(std::vector<Eigen::Vector3d> &points, const std::vector<double> &vars)
{

    const size_t  N = points.size();
    cv::Mat cloud_divergence(N, 1, CV_32FC3);
    cv::Mat cloud_convergence(N, 1, CV_32FC3);

    cv::Point3f* data_diver = cloud_divergence.ptr<cv::Point3f>();
    cv::Point3f* data_conver = cloud_convergence.ptr<cv::Point3f>();
    size_t n = 0, m = 0;
    for(size_t i = 0; i < N; ++i)
    {

        if(vars[i] < 0.01)
        {
            data_conver[n].x = points[i][0];
            data_conver[n].y = points[i][1];
            data_conver[n].z = points[i][2];
            n++;
        }
        else
        {
            data_diver[m].x = points[i][0];
            data_diver[m].y = points[i][1];
            data_diver[m].z = points[i][2];
            m++;
        }
    }
    cloud_divergence.resize(m);
    cloud_convergence.resize(n);

    {
        std::unique_lock<std::mutex> lock(mMutexCloud);
        clouds_.clear();
        colors_.clear();
        if(m > 0)
        {
            clouds_.push_back(cloud_divergence);
            colors_.push_back(cv::viz::Color(255.0, 255.0, 255.0));
        }
        if(n > 0)
        {
            clouds_.push_back(cloud_convergence);
            colors_.push_back(cv::viz::Color(0.0, .00, 255.0));
        }
    }

}