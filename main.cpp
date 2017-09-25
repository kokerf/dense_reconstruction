#include <iostream>

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
    DepthFilter depth_ilter;

    Sophus::SE3 Wc;
    Sophus::SE3 C2W;
    dataset[0].getPose(Wc);
    C2W = Wc.inverse();
    for (std::vector<DatasetEntry>::iterator data_itr = dataset.begin(); data_itr != dataset.end(); data_itr+=3)
    {
        cv::Mat image;
        data_itr->getImage(image, cv::IMREAD_GRAYSCALE);

        Sophus::SE3 pose;
        data_itr->getPose(pose);

        pose = C2W*pose;
        Frame::Ptr frame = std::make_shared<Frame>(Frame(image, 0, K, pose));

        depth_ilter.addFrame(frame);

        std::vector<Eigen::Vector3d> mpts;
        std::vector<double> vars;

        depth_ilter.getAllPoints(mpts, vars);

        pcl_viewer.update(mpts);

        cv::imshow("image", image);
        cv::waitKey(0);
    }



    return 0;
}