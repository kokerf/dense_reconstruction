#include "pcl_viewer.h"

PclViewer::PclViewer(const std::string name):
    pcl_viewer_(new pcl::visualization::CloudViewer(name))
{
}

void PclViewer::update(const std::vector<Eigen::Vector3d>& points)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>());

    for(std::vector<Eigen::Vector3d>::const_iterator mpt = points.begin(); mpt != points.end(); ++mpt)
    {

        pcl::PointXYZRGBA p;
        p.x = (*mpt)[0];
        p.y = (*mpt)[1];
        p.z = (*mpt)[2];

        p.r = 200;
        p.g = 200;
        p.b = 200;

        cloud->points.push_back(p);
    }

    update(cloud);
}

void PclViewer::update(const std::vector<Eigen::Vector3d> &points, const std::vector<double> &vars)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>());

    const int size = points.size();

    bool colored_flag = false;
    if(vars.size() == points.size())
        colored_flag = true;

    for(int i = 0; i < size; i++)
    {

        pcl::PointXYZRGBA p;
        p.x = points[i][0];
        p.y = -points[i][1];
        p.z = points[i][2];

        p.r = 200;
        p.g = 200;
        p.b = 200;
        if(colored_flag && vars[i] < 0.01)
        {
            p.r = 250;
            p.g = 0;
            p.b = 0;
        }

        cloud->points.push_back(p);
    }

    update(cloud);
}

void PclViewer::update(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud)
{
    pcl_viewer_->showCloud(cloud);
}