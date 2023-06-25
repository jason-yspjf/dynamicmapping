/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef POINTCLOUDMAPPING_H
#define POINTCLOUDMAPPING_H

#include "System.h"
//#include "YOLOv3SE.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <condition_variable>

// for clustering
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <iostream>

#include "yolo_v2_class.hpp"
#include "DynamicObjectDetecting.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <vector>
#include "FrameDrawer.h"


typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointNCloudT;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;
typedef pcl::PointCloud<pcl::PointXYZL> pointcloudL;


using namespace ORB_SLAM2;
//class FrameDrawer;



class PointCloudMapping
{
public:
    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
    enum cloudType
    {RAW = 0, FILTERED = 1, REMOVAL = 2, CLUSTER = 3};
    PointCloudMapping( double resolution_ );
    std::vector<cv::Scalar> colors;
    //void init(std::vector<std::string> specifiedThings, std::string cfgFile, std::string weightFile, std::string labelPath, double DetectTh);
    //std::vector<std::string> id2name;
    //std::unique_ptr<Detector> detector;
    // 插入一个keyframe，会更新一次地图
    void insertKeyFrame( KeyFrame* kf, cv::Mat& color, cv::Mat& depth, cv::Mat& colorDet );
    void shutdown();
    void viewer();
    void final_process();
    void projection();
    void CorOut();
    void generate_octomap(float resolution);
//    void filtCloud(float leaveSize = 0.014f);
//    void removePlane();
//    void cluster();

    //extern vector<cv::Mat> SavePic;
    
    //void myTry(FrameDrawer* mMapDrawer);
    //vector<cv::Mat> SaveRes;

    //YOLOv3 detector;
    cv::Mat dye_gray(cv::Mat &gray);
    PointCloud::Ptr ECE(PointCloud::Ptr cloud);
    PointCloud::Ptr cylinderSeg(PointCloud::Ptr cloud);
    PointCloud::Ptr regionGrowingSeg(PointCloud::Ptr cloud_in);
    PointCloud::Ptr colorRegionGrowingSeg(pcl::PointCloud <pcl::PointXYZRGB>::Ptr  cloud_in);

    PointCloud::Ptr CEC(const std::string& filename);
    void PointCloudXYZRGBAtoXYZ(const pcl::PointCloud<pcl::PointXYZRGBA>& in,
                            pcl::PointCloud<pcl::PointXYZ>& out);
    void PointXYZRGBAtoXYZ(const pcl::PointXYZRGBA& in,
                                pcl::PointXYZ& out);
    void PointXYZRGBtoXYZRGBA(const pcl::PointXYZRGB& in,
                                pcl::PointXYZRGBA& out);
    void PointCloudXYZRGBtoXYZRGBA(const pcl::PointCloud<pcl::PointXYZRGB>& in,
                            pcl::PointCloud<pcl::PointXYZRGBA>& out);
    void PointXYZLtoXYZ(const pcl::PointXYZL& in,
                   pcl::PointXYZ& out);
    void obj2pcd(const std::string& in, const std::string& out);
    void poisson_reconstruction(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr incloud);
    void cpf_seg(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr incloud);

    
protected:
    PointCloud::Ptr generatePointCloud(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, int i);
    PointCloud::Ptr generatePointCloudforDynamic(KeyFrame* kf, cv::Mat& color, cv::Mat& depth);
    PointCloud::Ptr globalMap;
    PointLCloudT::Ptr lableMap;
    shared_ptr<thread>  viewerThread;   
    
    bool    shutDownFlag    =false;
    mutex   shutDownMutex;  
    
    condition_variable  keyFrameUpdated;
    mutex               keyFrameUpdateMutex;

    vector<PointCloud::Ptr> Vecforpointcloud;
    PointCloud::Ptr globalMaprefine;
    uint16_t totalKeyframeSize =0;

    // data to generate point clouds
    vector<KeyFrame*>       keyframes;
    vector<cv::Mat>         colorImgs;
    vector<cv::Mat>         depthImgs;
    vector<cv::Mat>         colorDetImgs;
    mutex                   keyframeMutex;
    uint16_t                lastKeyframeSize =0;
    //add by me begin
//    pcl::SACSegmentation<PointT> seg;
//    pcl::PointIndices::Ptr inliers;
//    pcl::ModelCoefficients::Ptr coefficients;
//    pcl::ExtractIndices<PointT> extract;
//
//    pcl::VoxelGrid<PointT> vg;
//    pcl::PassThrough<PointT> pass;
//
//    PointCloud::Ptr pointCloud_raw;
//    PointCloud::Ptr pointCloud_filtered;
//    PointCloud::Ptr pointCloud_removal;
//    PointCloud::Ptr pointCloud_cluster;
//    PointCloud::Ptr pointCloud_plane;
//    int colorForCluster[20][3];

    //add by me end
    double resolution = 0.01;
    pcl::VoxelGrid<PointT>  voxel;
    pcl::StatisticalOutlierRemoval<PointT> sor;// 创建滤波器对象

/*private:
    void loadLabel();
    // config parameters
    std::string mCfgFile, mWeightFile;
    std::string mLabelPath;
    double mDetectTh;
    std::vector<std::string> mSpecifiedThings;*/


};

#endif // POINTCLOUDMAPPING_H
