#include <iostream>
#include <math.h>

#include <opencv2/opencv.hpp>

#include "config.h"
#include "frame.h"
#include "data_reader.h"
#include "pcl_viewer.h"
#include "depth_filter.h"

std::string Config::FileName;

int main(int argc, char const *argv[])
{

    if (argc != 3) {
        std::cout << "Usge: ./dense_mapping datasetpath configflie" << std::endl;
    }

    std::string dataset_path(argv[1]);
    std::string dataset_sequence_file = dataset_path + "/first_200_frames_traj_over_table_input_sequence.txt";
    DatasetReader datasetReader(dataset_path, dataset_sequence_file);

    Config::FileName = std::string(argv[2]);
    cv::Mat K = Config::cameraK();
    int delay = (int)(1.0 / Config::cameraFps());


    std::vector<DatasetEntry> dataset;
    bool succeed = datasetReader.readDataSequence(dataset);

    if(!succeed) {
        return -1;
    }

    PclViewer pcl_viewer("mapping");
    DepthFilter depth_filter;

    Sophus::SE3 T_r_w;
    Sophus::SE3 C2W;
    dataset[0].getPose(T_r_w);
    C2W = T_r_w.inverse();

    for (std::vector<DatasetEntry>::iterator data_itr = dataset.begin(); data_itr != dataset.end(); data_itr+=2)
    {
        cv::Mat image;
        if(!data_itr->getImage(image, cv::IMREAD_GRAYSCALE))
            break;

        Sophus::SE3 pose;
        if(!data_itr->getPose(pose))
            break;

        //! 数据集就是这样的
        pose = pose.inverse() * T_r_w;

        Frame::Ptr frame = std::make_shared<Frame>(Frame(image, 0, K, pose));

        depth_filter.addFrame(frame);

        std::vector<Eigen::Vector3d> mpts;
        std::vector<double> vars;

        depth_filter.getAllPoints(mpts, vars);

        pcl_viewer.update(mpts, vars);

        cv::imshow("image", image);
        cv::imshow("gradx", frame->getGradxInLevel(0));
        cv::imshow("grady", frame->getGradyInLevel(0));
        cv::waitKey(0);
    }


    return 0;
}