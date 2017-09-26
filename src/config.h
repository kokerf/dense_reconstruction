#ifndef DENSE_RECONSTRUCTION_CONFIG_H
#define DENSE_RECONSTRUCTION_CONFIG_H

#include <iostream>
#include <string>

#include <opencv2/core.hpp>


using std::string;

class Config
{
public:

    static cv::Mat cameraK(){return getInstance().K;}

    static cv::Mat cameraInvK(){return getInstance().invK;}

    static int imageWidth(){return getInstance().width;}

    static int imageHeight(){return getInstance().height;}

    static double cameraFps(){return getInstance().fps;}

    static int seedMinGrad(){return getInstance().seed_min_grad;}

    static int seedInitVar2(){return getInstance().seed_init_var2;}

    static double minEplGrad2(){return getInstance().depth_fliter_minEplGrad2;}

    static double minEplAngle2(){return getInstance().depth_fliter_minEplAngle2;}

    static double minNCCScore(){return getInstance().depth_fliter_minNCCScore;}

    static double pixelError(){return getInstance().depth_fliter_pixelError;}


private:
    static Config& getInstance()
    {
        static Config instance(FileName);
        return instance;
    }

    Config(string& file_name)
    {
        cv::FileStorage fs(file_name.c_str(), cv::FileStorage::READ);
        if(!fs.isOpened())
        {
            std::cerr << "Failed to open settings file at: " << file_name << std::endl;
            exit(-1);
        }

        //! camera parameters
        fx = (double)fs["Camera.fx"];
        fy = (double)fs["Camera.fy"];
        cx = (double)fs["Camera.cx"];
        cy = (double)fs["Camera.cy"];

        K = cv::Mat::eye(3,3,CV_64F);
        K.at<double>(0,0) = fx;
        K.at<double>(1,1) = fy;
        K.at<double>(0,2) = cx;
        K.at<double>(1,2) = cy;

        invK = K.inv();

        width = (int)fs["Camera.width"];
        height = (int)fs["Camera.height"];

        fps = (double)fs["Camera.fps"];

        seed_min_grad = (int)fs["Seed.minGrad"];
        seed_init_var2 = (double)fs["Seed.initVar"];
        seed_init_var2 *= seed_init_var2;

        depth_fliter_minEplGrad2 = (double)fs["DepthFilter.minEplGrad"];
        depth_fliter_minEplGrad2*= depth_fliter_minEplGrad2;

        depth_fliter_minEplAngle2 = (double)fs["DepthFilter.minEplAngle"];
        depth_fliter_minEplAngle2 *= depth_fliter_minEplAngle2;

        depth_fliter_minNCCScore = (double)fs["DepthFilter.minNCCScore"];

        depth_fliter_pixelError = (double)fs["DepthFilter.pixelError"];

        fs.release();
    }

public:
    //! config file's name
    static string FileName;

private:
    //! camera parameters
    double fx, fy, cx, cy;

    cv::Mat K;
    cv::Mat invK;

    int width;
    int height;

    double fps;

    int seed_min_grad;
    double seed_init_var2;

    double depth_fliter_minEplGrad2;
    double depth_fliter_minEplAngle2;
    double depth_fliter_minNCCScore;
    double depth_fliter_pixelError;

};


#endif //DENSE_RECONSTRUCTION_CONFIG_H
