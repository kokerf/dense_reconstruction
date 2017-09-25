#ifndef DENSE_RECONSTRUCTION_DATA_READE_H
#define DENSE_RECONSTRUCTION_DATA_READE_H

#include <fstream>
#include <vector>
#include <string>

#include <opencv2/core.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sophus/se3.h>
using Sophus::SE3;

class DatasetEntry{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    DatasetEntry(const std::string& image_file_name, const Eigen::Vector3d& translation, const Eigen::Quaterniond& quaternion):
            image_file_name_(image_file_name), translation_(translation), quaternion_(quaternion)
    {}

    bool getImage(cv::Mat& img, int read_type = cv::IMREAD_UNCHANGED) const
    {
        img = cv::imread(image_file_name_, read_type);
        return !img.empty();
    }

    bool getPose(SE3& T) const
    {
        T = SE3(quaternion_, translation_);
        return true;
    }

private:
    const std::string image_file_name_;
    const Eigen::Vector3d translation_;
    const Eigen::Quaterniond quaternion_;
};

class DatasetReader{
public:
    DatasetReader(const std::string& dataset_path, std::string& sequence_file):
            dataset_path_(dataset_path), sequence_file_(sequence_file)
    {}

    bool readDataSequence(std::vector<DatasetEntry>& dataset)
    {
        if(dataset_path_.empty() || sequence_file_.empty())
        {
            return false;
        }

        dataset.clear();

        std::ifstream sequence_file_str(sequence_file_);
        if(!sequence_file_str.is_open())
            return false;

        std::string line;
        while(getline(sequence_file_str, line))
        {
            std::stringstream line_str(line);
            std::string imgFileName;
            Eigen::Vector3d translation;
            Eigen::Quaterniond quaternion;
            line_str >> imgFileName;
            line_str >> translation.x();
            line_str >> translation.y();
            line_str >> translation.z();
            line_str >> quaternion.x();
            line_str >> quaternion.y();
            line_str >> quaternion.z();
            line_str >> quaternion.w();

            dataset.push_back(DatasetEntry(dataset_path_+"/images/"+imgFileName, translation, quaternion));
        }

        sequence_file_str.close();

        return true;
    }


private:
    const std::string dataset_path_;
    const std::string sequence_file_;

};


#endif //DENSE_RECONSTRUCTION_DATA_READE_H
