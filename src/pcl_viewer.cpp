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

void PclViewer::update(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud)
{
    pcl_viewer_->showCloud(cloud);
}