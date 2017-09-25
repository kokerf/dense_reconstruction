#ifndef DENSE_RECONSTRUCTION_PCL_VIEWER_H
#define DENSE_RECONSTRUCTION_PCL_VIEWER_H

#include <Eigen/Core>
#include <pcl/visualization/cloud_viewer.h>


class PclViewer
{
public:
    PclViewer(const std::string name);

    void update(const std::vector<Eigen::Vector3d>& points);

    void update(const pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud);

    typedef std::shared_ptr<PclViewer> Ptr;

private:

    pcl::visualization::CloudViewer* pcl_viewer_;

};


#endif //DENSE_RECONSTRUCTION_PCL_VIEWER_H
