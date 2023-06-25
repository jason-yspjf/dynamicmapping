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
#include "pointcloudmapping.h"
//#include "YOLOv3SE.h"
#include "segmentation.h"
#include <Eigen/Geometry>
#include <KeyFrame.h>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <ctime>
#include <pcl/surface/gp3.h>

#include <pcl/surface/poisson.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/poisson.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/projection_matrix.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "Converter.h"
#include <boost/make_shared.hpp>

#include <pcl/filters/project_inliers.h>

/*#ifdef GPU
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#endif*/

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <unistd.h>
#include <time.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>

//VTK include needed for drawing graph lines
#include <vtkPolyLine.h>

#include <assert.h>
#include <octomap/octomap.h>
#include <octomap/ColorOcTree.h>
#include "yolo_v2_class.hpp"
#include "DynamicObjectDetecting.h"
#include "FrameDrawer.h"





using namespace cv;
using namespace std;
using namespace pcl;
using namespace APC;
using namespace pcl::io;
using namespace pcl::console;
typedef pcl::PointXYZI PointTypeIO;
typedef pcl::PointXYZINormal PointTypeFull;


int pub_port = 6666;
#define NUM 5


int k = 500;
int min_size = 500;


/*template<typename T, typename... Ts>
std::unique_ptr<T> make_unique(Ts&&... params)
{
    return std::unique_ptr<T>(new T(std::forward<Ts>(params)...));
}*/



PointCloudMapping::PointCloudMapping(double resolution_) {

    this->resolution = resolution_;
    voxel.setLeafSize(resolution, resolution, resolution);
    globalMap = boost::make_shared<PointCloud>();
    globalMaprefine = boost::make_shared<PointCloud>();
    viewerThread = make_shared<thread>(bind(&PointCloudMapping::viewer, this));

}

/*void PointCloudMapping::init(std::vector<std::string> specifiedThings, std::string cfgFile, std::string weightFile, std::string labelPath, double detectTh)
{

    mCfgFile = cfgFile;
    mSpecifiedThings = specifiedThings;
    mWeightFile = weightFile;
    mLabelPath = labelPath;
    mDetectTh = detectTh;
    detector = ::make_unique<Detector>(mCfgFile, mWeightFile,0); //初始化检测器, 第三个参数是gpu_id
    loadLabel();
}

void PointCloudMapping::loadLabel()
{
    std::ifstream fLabel; 
    id2name.resize(80);

    fLabel.open(mLabelPath.c_str());
    int i = 0;
    while (!fLabel.eof()) {
        std::string name;
        getline(fLabel,name);
        if (!name.empty()) {
            id2name[i++] = name;
        }
    }
}*/


void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}


void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, cv::Mat& colorDet)
{


    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;

    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back( kf );


    colorImgs.push_back( color.clone());

    depthImgs.push_back( depth.clone() );


    colorDetImgs.push_back( colorDet.clone() );

    keyFrameUpdated.notify_one();
}




pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame* keyf, cv::Mat& color, cv::Mat& depth, int i)
{
    std::cout << "Color picture: " << color.rows << " " << color.cols << std::endl;
    std::cout << "Depth picture: " << depth.rows << " " << depth.cols << std::endl;
    PointCloud::Ptr tmp( new PointCloud() );
    // point cloud is null ptr
    for ( int m=0; m<depth.rows; m+=1 )
    {
        for ( int n=0; n<depth.cols; n+=1 )
        {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d > 5)
                continue;
            //点云选择
            /*if(color.ptr<uchar>(m)[n*3] == 0 && color.ptr<uchar>(m)[n*3+1] == 0 && color.ptr<uchar>(m)[n*3+2] == 0) {
                continue;
            }*/
            PointT p;
            p.z = d;
            p.x = ( n - keyf->cx) * p.z / keyf->fx;
            p.y = ( m - keyf->cy) * p.z / keyf->fy;

            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];

//            if(p.b == p.g || p.b == p.r)
//                continue;
            if(p.b == p.g && p.b == p.r) {
                continue;
            }

            tmp->points.push_back(p);
        }
    }

    PointCloud::Ptr cloud1 (new PointCloud);
    pcl::RadiusOutlierRemoval<pcl::PointXYZRGBA> pcFilter;
    pcFilter.setInputCloud(tmp);
    pcFilter.setRadiusSearch(0.05);
    pcFilter.setMinNeighborsInRadius(5);
    pcFilter.filter(*cloud1);

    Vecforpointcloud[i] = cloud1;

    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( keyf->GetPose() );
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud( *cloud1, *cloud, T.inverse().matrix()); //变换矩阵，tmp保存了从彩色图和深度图直接获得的点云，cloud为最终输出的点云
    cloud->is_dense = false;

    
    cout<<"generate point cloud for kf "<<keyf->mnId<<", size="<<cloud->points.size()<<endl;
    return cloud;
}


pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloudforDynamic(KeyFrame* keyf, cv::Mat& color, cv::Mat& depth)
{
    std::cout << "Color picture: " << color.rows << " " << color.cols << std::endl;
    std::cout << "Depth picture: " << depth.rows << " " << depth.cols << std::endl;
    PointCloud::Ptr tmp( new PointCloud() );
    // point cloud is null ptr
    for ( int m=0; m<depth.rows; m+=3 )
    {
        for ( int n=0; n<depth.cols; n+=3 )
        {
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d > 5)
                continue;
            //点云选择
            /*if(color.ptr<uchar>(m)[n*3] == 0 && color.ptr<uchar>(m)[n*3+1] == 0 && color.ptr<uchar>(m)[n*3+2] == 0) {
                continue;
            }*/
            PointT p;
            p.z = d;
            p.x = ( n - keyf->cx) * p.z / keyf->fx;
            p.y = ( m - keyf->cy) * p.z / keyf->fy;

            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];

//            if(p.b == p.g || p.b == p.r)
//                continue;
            tmp->points.push_back(p);
        }
    }

    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( keyf->GetPose() );
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix()); //变换矩阵，tmp保存了从彩色图和深度图直接获得的点云，cloud为最终输出的点云
    cloud->is_dense = false;

    PointCloud::Ptr cloud1 (new PointCloud);
    pcl::RadiusOutlierRemoval<pcl::PointXYZRGBA> pcFilter;
    pcFilter.setInputCloud(cloud);
    pcFilter.setRadiusSearch(0.05);
    pcFilter.setMinNeighborsInRadius(5);
    pcFilter.filter(*cloud1);

    cout<<"generate point cloud for kf "<<keyf->mnId<<", size="<<cloud1->points.size()<<endl;
    return cloud1;
}



//void PointCloudMapping::viewer()
//{
//    pcl::visualization::CloudViewer viewer("viewer");
//    while(1)
//    {
//        {
//            unique_lock<mutex> lck_shutdown( shutDownMutex );
//            if (shutDownFlag)
//            {
//                break;
//            }
//        }
//        {
//            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
//            keyFrameUpdated.wait( lck_keyframeUpdated );
//        }
//
//        // keyframe is updated
//        size_t N=0;
//        {
//            unique_lock<mutex> lck( keyframeMutex );
//            N = keyframes.size();
//        }
//
//        for ( size_t i=lastKeyframeSize; i<N ; i++ )
//        {
//            PointCloud::Ptr prep = generatePointCloud( keyframes[i], colorImgs[i], depthImgs[i] );
//            PointCloud::Ptr p = regionGrowingSeg(prep);
//            *globalMap += *p;
//        }
//        //PointCloud::Ptr tmp(new PointCloud());
//        voxel.setInputCloud( globalMap );
////        voxel.filter( *tmp );
////        globalMap->swap( *tmp );
//        viewer.showCloud( globalMap );
//        cout << "show global map, size=" << globalMap->points.size() << endl;
//        lastKeyframeSize = N;
//    }
//
// }

/*void PointCloudMapping::myTry(FrameDrawer* mMapDrawer)
{
    int n = mMapDrawer->mSavePic.size();
    for(int i = 0; i < n; i++) {
        SaveRes.push_back(mMapDrawer->mSavePic[i]);
    }
}*/


void PointCloudMapping::viewer()
{
    /*std::vector<cv::Scalar> colors;
    string line, number;
    ifstream f("color.txt");
    if (!f.good())
    {
        cout << "Cannot open file" << endl;
        exit(-1);
    }



    vector<int> v;
    while(std::getline(f, line))
    {
        istringstream is(line);

        while(std::getline(is, number, ','))
        {
            v.push_back(atoi(number.c_str()));
        }
        colors.push_back(cv::Scalar(v[0],v[1],v[2]));
        v.clear();
    }*/
//    for (int i = 0; i < 80; i++) {
//        colors.push_back(cv::Scalar(rand() % 127 + 128, rand() % 127 + 128, rand() % 127 + 128));
//    }

    //detector.Create("yolov3.weights", "yolov3.cfg", "coco.names");
    //sleep(3);
    pcl::visualization::CloudViewer viewer("viewer");


    while(1)
    {
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }

        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }


        // keyframe is updated
        size_t N=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N = keyframes.size();
        }

        Vecforpointcloud.resize(N);
        totalKeyframeSize = N;


        {

            for (size_t i=lastKeyframeSize; i<N ; i++) {

                /*cv::Mat tmp_color = colorImgs[i];
                char img[20];
                sprintf(img, "%s%d%s", "img/image", i, ".png");
                cv::imwrite(img,tmp_color);
                //PointCloud::Ptr pre_p = generatePointCloud(keyframes[i], colorImgs[i], depthImgs[i]);
                //PointCloud::Ptr p = regionGrowingSeg(pre_p);
                //*globalMap += *pre_p;
                //sleep(3);
                cv::Mat img_tmp_color = cv::imread(img);
                //std::vector<BoxSE> boxes = detector.Detect(img_tmp_color, 0.5F);
                std::vector<bbox_t> results = detector->detect(img_tmp_color, mDetectTh); 
                sleep(3);
                //continue;
                int n = results.size();
                for (int i = 0; i < n; i++) {
                    if(id2name[results[i].obj_id] ==  "person")
                    {
                        cv::rectangle(img_tmp_color, cv::Rect(results[i].x, results[i].y, results[i].w, results[i].h), colors[80], -1, 4);//-1代表在目标框内进行填充
                    }
                    //cv::putText(img, detector.Names(box.m_class), box.tl(), cv::FONT_HERSHEY_SIMPLEX, 1.0, colors[box.m_class], 2);
                    //cv::rectangle(img, box, colors[box.m_class], 2);
                    cv::rectangle(img_tmp_color, cv::Rect(results[i].x, results[i].y, results[i].w, results[i].h), colors[results[i].obj_id], -1, 4);
                }

                char img_det[25];
                sprintf(img_det, "%s%d%s", "img_det/image", i, ".png");
                cv::imwrite(img_det,img_tmp_color);*/

                /*char img[25];
                sprintf(img, "%s%d%s", "img_det/image", i , ".png");
                cv::Mat img_tmp_color = cv::imread(img);*/
                
                //cv::Mat img_tmp_color = SaveRes[i];

                // PointCloud::Ptr surf_pdyn = generatePointCloudforDynamic(keyframes[i], colorImgs[i], depthImgs[i]);


                // std::stringstream ss;
                // ss << "cloud/cloud_keyframe_" << i << ".pcd";
                // pcl::PCDWriter pcdwriter;
                // pcdwriter.write<pcl::PointXYZRGBA>(ss.str(), *surf_pdyn);

                PointCloud::Ptr surf_p = generatePointCloud(keyframes[i], colorImgs[i], depthImgs[i], i);
                //PointCloud::Ptr p = RegionGrowingSeg(surf_p);
                
                *globalMap += *surf_p;

            }


        }

        voxel.setInputCloud( globalMap );

        viewer.showCloud( globalMap );
//      pcl::PointCloud<pcl::PointXYZ> ply_file;
//      PointCloudXYZRGBAtoXYZ(*globalMap,ply_file);

        cout << "show global map, size=" << globalMap->points.size() << endl;
        lastKeyframeSize = N;

        {
            for(size_t i = 0; i < totalKeyframeSize; i++) {
                Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( keyframes[i]->GetPose() );
                PointCloud::Ptr cloudforrefine(new PointCloud);
                pcl::transformPointCloud( *Vecforpointcloud[i], *cloudforrefine, T.inverse().matrix()); //变换矩阵，tmp保存了从彩色图和深度图直接获得的点云，cloud为最终输出的点云
                cloudforrefine->is_dense = false;
                *globalMaprefine += *cloudforrefine;
            }
            pcl::RadiusOutlierRemoval<pcl::PointXYZRGBA> pcFilter;
            pcFilter.setInputCloud(globalMaprefine);
            pcFilter.setRadiusSearch(0.05);
            pcFilter.setMinNeighborsInRadius(5);
            pcFilter.filter(*globalMaprefine);
            pcl::PCDWriter pcdwriter;
            pcdwriter.write<pcl::PointXYZRGBA>("global_color-refine.pcd", *globalMaprefine);
            pcl::io::savePLYFileASCII("global_color-refine.ply", *globalMaprefine);
        }

    }

    


    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PCDWriter pcdwriter;
    pcdwriter.write<pcl::PointXYZRGBA>("global_color.pcd", *globalMap);//write global point cloud map and save to a pcd file
    // pcdwriter.write<pcl::PointXYZRGBA>("global_color-refine.pcd", *globalMaprefine);
    pcl::io::savePLYFileASCII("global_color.ply", *globalMap);
    // pcl::io::savePLYFileASCII("global_color-refine.ply", *globalMaprefine);
    //cpf_seg(globalMap);
    //poisson_reconstruction(globalMap);
    //detector.Release();
    //final_process();
    /*projection();*/
    /*CorOut();*/
    generate_octomap(0.01);
    /*
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PCDWriter pcdwriter;
    pcdwriter.write<pcl::PointXYZRGBA>("global.pcd", *globalMap);//write global point cloud map and save to a pcd file
    pcl::io::savePCDFile("pointcloud.pcd", *globalMap );


    pcl::PointCloud<pcl::PointXYZ> ply_file;
    PointCloudXYZRGBAtoXYZ(*globalMap,ply_file);
    pcl::PLYWriter plywriter;
    plywriter.write<pcl::PointXYZ>("test.ply",ply_file,true);

     */
    
    exit(0);


}

void PointCloudMapping::generate_octomap(float resolution)
{
    pcl::PointCloud<pcl::PointXYZRGBA> cloud;
    pcl::io::loadPCDFile<pcl::PointXYZRGBA> ("global_color-refine.pcd", cloud);

    octomap::ColorOcTree tree( resolution );
    for (auto p:cloud.points)
    {
        // 将点云里的点插入到octomap中
        tree.updateNode( octomap::point3d(p.x, p.y, p.z), true );
    }

    for (auto p:cloud.points)
    {
        tree.integrateNodeColor( p.x, p.y, p.z, p.r, p.g, p.b );
    }

    // 更新octomap
    tree.updateInnerOccupancy();
    // 存储octomap, 注意要存成.ot文件而非.bt文件
    tree.write("final_semantic_octomap.ot");
    cout << "finish generating octomap, see details in final_semantic_octomap.ot." << endl;
}


void PointCloudMapping::final_process()
{
    vector<int> vcup;
    vector<int> vteddy;
    vector<int> vkeyboard;
    vector<int> vmouse;
    vector<int> vmonitor;
    vector<int> vbook;


    PointCloudT::Ptr tgt (new PointCloudT);
    pointcloudL::Ptr ll (new pointcloudL);

    pcl::io::loadPCDFile<pcl::PointXYZRGBA>("global_color.pcd",*tgt);
    pcl::io::loadPCDFile<pcl::PointXYZL>("segmentation.pcd",*ll);

    PointCloudT Final1 = *tgt;
    pcl::PointCloud<pcl::PointXYZL> lcloud = *ll;
    //cout <<lcloud.size()<<endl;
    //cout<<"1:"<<Final2.points.size()<<"/n"<<"2:"<<Final2.points.size();
    
    map<int, int> mapmonitor;
    map<int, int> mapmouse;
    map<int, int> mapkeyboard;
    map<int, int> mapcup;

    for(int i=0;i< Final1.points.size();i++)

    {

        pcl::PointXYZRGBA basic_point;
        pcl::PointXYZL basic_pointl;
        basic_point.x = Final1.points[i].x;
        basic_point.y = Final1.points[i].y;
        basic_point.z = Final1.points[i].z;
        basic_pointl.x = lcloud.points[i].x;
        basic_pointl.y = lcloud.points[i].y;
        basic_pointl.z = lcloud.points[i].z;
        basic_point.rgba = Final1.points[i].rgba;
        uint32_t rgba = Final1.points[i].rgba;
        int r =rgba >> 16 & 0x0000ff;
        int g = rgba >> 8 & 0x0000ff;
        int b = rgba & 0x0000ff;
        //cout <<basic_point.rgba<<endl;

        int prelabelmonitor;
        int prelabelmouse;
        int prelabelkeyboard;
        int prelabelcup;


        if(b==0&&g==252&&r==124){//0,252,124
            //cout <<r<<endl;//monitor
            prelabelmonitor = lcloud[i].label;
            map<int, int>::iterator it = mapmonitor.find(prelabelmonitor);
            if(it != mapmonitor.end()) {
                it->second++;
            }
            else {
                mapmonitor.insert(make_pair(prelabelmonitor, 1));
            }
            vmonitor.push_back(i);
            lcloud[i].label = 300;

        }
        if(b==203&&g==192&&r==255){//203,192,255
            //cout <<r<<endl;//mouse
            prelabelmouse = lcloud[i].label;
            map<int, int>::iterator it = mapmouse.find(prelabelmouse);
            if(it != mapmouse.end()) {
                it->second++;
            }
            else {
                mapmouse.insert(make_pair(prelabelmouse, 1));
            }
            vmouse.push_back(i);
            lcloud[i].label = 400;

        }
        if(b==128&&g==249&&r==237){//128,249,237
            //cout <<r<<endl;//keyboard
            prelabelkeyboard = lcloud[i].label;
            map<int, int>::iterator it = mapkeyboard.find(prelabelkeyboard);
            if(it != mapkeyboard.end()) {
                it->second++;
            }
            else {
                mapkeyboard.insert(make_pair(prelabelkeyboard, 1));
            }
            vkeyboard.push_back(i);
            lcloud[i].label = 500;

        }
        if(b==50&&g==205&&r==50){//50,205,50
            //cout <<r<<endl;//teddy bear
            vteddy.push_back(i);
            lcloud[i].label = 600;


        }
        if(b==240&&g==160&&r==120){//240,160,120
            //cout <<r<<endl;//cup
            prelabelcup = lcloud[i].label;
            map<int, int>::iterator it = mapcup.find(prelabelcup);
            if(it != mapcup.end()) {
                it->second++;
            }
            else {
                mapcup.insert(make_pair(prelabelcup, 1));
            }
            vcup.push_back(i);
            lcloud[i].label = 700;

        }

        if(b==255&&g==228&&r==225){//50,205,50
            //cout <<r<<endl;//book
            vbook.push_back(i);
            lcloud[i].label = 800;
        }

    }

    int premaxlabelmonitorcount = 0;
    int premaxlabelmonitor = 0;
    for(map<int, int>::iterator cit = mapmonitor.begin(); cit != mapmonitor.end(); cit++) {
        premaxlabelmonitor = (cit->second > premaxlabelmonitorcount) ? cit->first : premaxlabelmonitor;
        premaxlabelmonitorcount = (cit->second > premaxlabelmonitorcount) ? cit->second : premaxlabelmonitorcount;
    }
    cout << "premaxlabelmonitor " << premaxlabelmonitor << " premaxlabelmonitorcount " << premaxlabelmonitorcount << endl;
    cout << "vmonitor.size() " << vmonitor.size() << endl;


    int premaxlabelmousecount = 0;
    int premaxlabelmouse = 0;
    for(map<int, int>::iterator cit = mapmouse.begin(); cit != mapmouse.end(); cit++) {
        premaxlabelmouse = (cit->second > premaxlabelmousecount) ? cit->first : premaxlabelmouse;
        premaxlabelmousecount = (cit->second > premaxlabelmousecount) ? cit->second : premaxlabelmousecount;
    }
    cout << "premaxlabelmouse " << premaxlabelmouse << " premaxlabelmousecount " << premaxlabelmousecount << endl;
    cout << "vmouse.size() " << vmouse.size() << endl;

    int premaxlabelkeyboardcount = 0;
    int premaxlabelkeyboard = 0;
    for(map<int, int>::iterator cit = mapkeyboard.begin(); cit != mapkeyboard.end(); cit++) {
        premaxlabelkeyboard = (cit->second > premaxlabelkeyboardcount) ? cit->first : premaxlabelkeyboard;
        premaxlabelkeyboardcount = (cit->second > premaxlabelkeyboardcount) ? cit->second : premaxlabelkeyboardcount;
    }
    cout << "premaxlabelkeyboard " << premaxlabelkeyboard << " premaxlabelkeyboardcount " << premaxlabelkeyboardcount << endl;
    cout << "vkeyboard.size() " << vkeyboard.size() << endl;

    int premaxlabelcupcount = 0;
    int premaxlabelcup = 0;
    for(map<int, int>::iterator cit = mapcup.begin(); cit != mapcup.end(); cit++) {
        premaxlabelcup = (cit->second > premaxlabelcupcount) ? cit->first : premaxlabelcup;
        premaxlabelcupcount = (cit->second > premaxlabelcupcount) ? cit->second : premaxlabelcupcount;
    }
    cout << "premaxlabelcup " << premaxlabelcup << " premaxlabelcupcount " << premaxlabelcupcount <<endl;
    cout << "vcup.size() " << vcup.size() << endl;

    for(int i = 0; i < lcloud.points.size(); i++) {
        if(premaxlabelmonitorcount * 2 > vmonitor.size() && lcloud[i].label == premaxlabelmonitor) {
            lcloud[i].label = 300;
        }
        if(premaxlabelmousecount * 1.8 > vmouse.size() && lcloud[i].label == premaxlabelmouse) {
            lcloud[i].label = 400;
        }
        if(premaxlabelkeyboardcount * 2 > vkeyboard.size() && lcloud[i].label == premaxlabelkeyboard) {
            lcloud[i].label = 500;
        }
        if(premaxlabelcupcount * 1.8 > vcup.size() && lcloud[i].label == premaxlabelcup) {
            lcloud[i].label = 700;
        }
    }

    pcl::PCDWriter pcdwriter;
    pcdwriter.write<pcl::PointXYZL>("semantic_map.pcd", lcloud);//write global
    cout <<"finish semantic labelling,please see details in semantic_map.pcd"<<endl;

}

void PointCloudMapping::CorOut()
{
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::io::loadPCDFile("semantic_map.pcd", *cloud);
    std::fstream fs;
	//读写文件
	fs.open("./11.txt",std::fstream::out);
	//打开文件路径，没有则创建文件
	for (size_t i = 0; i < cloud->points.size(); ++i) //points.size表示点云数据的大小
	{
	fs << cloud->points[i].x << "\t"
		//写入数据
		<< cloud->points[i].y << "\t" << cloud->points[i].z << "\t" << cloud->points[i].label << "\n";
	}
	fs.close();//关闭

}

void PointCloudMapping::projection()
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::io::loadPCDFile("global_color.pcd", *cloud);
	//定义模型系数对象coefficients并填充数据
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	//参数模型为 ax+by+cz+d=0
	// z=0 即为x-y得一个平面
	coefficients->values.resize(4);
	coefficients->values[0] = 0;
	coefficients->values[1] = 0;
	coefficients->values[2] = 1.0;
	coefficients->values[3] = 0;
	//创建投影滤波对象
	pcl::ProjectInliers<pcl::PointXYZRGBA> proj;
	proj.setModelType(pcl::SACMODEL_PLANE);
	proj.setInputCloud(cloud);
	proj.setModelCoefficients(coefficients);
	proj.filter(*cloud2);
	pcl::PCDWriter pcdwriter;
    pcdwriter.write<pcl::PointXYZRGBA>("projection_result.pcd", *cloud2);//write global
    cout <<"finish projecting,please see details in projection_result.pcd"<<endl;
    
    pcl::PointCloud<pcl::PointXYZRGBA> cloud3 = *cloud2;
    float xsum = 0, ysum = 0;
    int size = cloud3.points.size();
    cout << size << endl;
    for(int i = 0; i < size; i++) {
        xsum += cloud3.points[i].x;
        ysum += cloud3.points[i].y;
    }
    float xave = xsum / size;
    float yave = ysum / size;
    cout << xave << " " << yave << endl;
    float boudingxmin = xave - 1.5;
    float boudingxmax = xave + 1.5;
    float boudingymin = yave - 1.5;
    float boudingymax = yave + 1.5;
    cout << boudingxmin << " " << boudingxmax << " " << boudingymin << " " << boudingymax << endl;
    int row = 600;
    int col = 600;
    Mat Image = Mat(row, col, CV_8UC3);
    for(int i = 0; i < row; i++) {
        for(int j = 0; j < col; j++) {
            Image.at<cv::Vec3b>(i, j)[0] = 0;
            Image.at<cv::Vec3b>(i, j)[1] = 0;
            Image.at<cv::Vec3b>(i, j)[2] = 0;
        }
    }
    int imgorix = boudingxmin * 200;
    int imgoriy = boudingymin * 200;
    cout << imgorix << " " << imgoriy << endl;
    std::fstream fs;
    fs.open("./22.txt",std::fstream::out);
    for(int i = 0; i < size; i++) {
        if(cloud3.points[i].x < boudingxmin || cloud3.points[i].x > boudingxmax || cloud3.points[i].y < boudingymin || cloud3.points[i].y > boudingymax) {
            continue;
        }
        int u = cloud3.points[i].x * 200 - imgorix;
        int v = cloud3.points[i].y * 200 - imgoriy;
        fs << u << "\t" << v << "\n";
        if(u < 0 || u >= 600 || v < 0 || v >= 600) {
            continue;
        }
        uint32_t rgba = cloud3.points[i].rgba;
        int r =rgba >> 16 & 0x0000ff;
        int g = rgba >> 8 & 0x0000ff;
        int b = rgba & 0x0000ff;
        Image.at<cv::Vec3b>(u, v)[0] = b;
        Image.at<cv::Vec3b>(u, v)[1] = g;
        Image.at<cv::Vec3b>(u, v)[2] = r;
    }
    fs.close();
    imwrite("projection.jpg", Image);
    cout <<"finish projecting,please see details in projection.jpg"<<endl;
}

void PointCloudMapping::obj2pcd(const std::string& inputFilename, const std::string& outputFilename)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;

    // Input stream
    std::ifstream is(inputFilename.c_str());

    // Read line by line
    for(std::string line; std::getline(is, line); )
    {
        std::istringstream in(line);

        std::string v;
        in >> v;
        if (v != "v") continue;

        // Read x y z
        float x, y, z;
        in >> x >> y >> z;
        cloud.push_back(pcl::PointXYZ(x, y, z));
    }

    is.close();

    // Save to pcd file
    pcl::io::savePCDFileBinaryCompressed(outputFilename, cloud);
}


void PointCloudMapping::PointXYZRGBAtoXYZ(const pcl::PointXYZRGBA& in,
                                pcl::PointXYZ& out)
{
    out.x = in.x; out.y = in.y; out.z = in.z;
}

void PointCloudMapping::PointXYZLtoXYZ(const pcl::PointXYZL& in,
                                          pcl::PointXYZ& out)
{
    out.x = in.x; out.y = in.y; out.z = in.z;

}

void PointCloudMapping::PointCloudXYZRGBAtoXYZ(const pcl::PointCloud<pcl::PointXYZRGBA>& in,
                            pcl::PointCloud<pcl::PointXYZ>& out)
{
    out.width = in.width;
    out.height = in.height;
    for (size_t i = 0; i < in.points.size(); i++)
    {
        pcl::PointXYZ p;
        PointXYZRGBAtoXYZ(in.points[i],p);
        out.points.push_back (p);
    }
}



void PointCloudMapping::PointXYZRGBtoXYZRGBA(const pcl::PointXYZRGB& in,
                                pcl::PointXYZRGBA& out)
{
    out.x = in.x; out.y = in.y; out.z = in.z;
    out.r = in.r; out.g = in.g; out.b = in.z; out.a =0;

}

void PointCloudMapping::PointCloudXYZRGBtoXYZRGBA(const pcl::PointCloud<pcl::PointXYZRGB>& in,
                            pcl::PointCloud<pcl::PointXYZRGBA>& out)
{
    out.width = in.width;
    out.height = in.height;
    for (size_t i = 0; i < in.points.size(); i++)
    {
        pcl::PointXYZRGBA p;
        PointXYZRGBtoXYZRGBA(in.points[i],p);
        out.points.push_back (p);
    }
}


void PointCloudMapping::poisson_reconstruction(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr object_cloud)
{

    pcl::PointCloud<PointXYZRGB>::Ptr cloud(new pcl::PointCloud<PointXYZRGB>());
    pcl::copyPointCloud(*object_cloud, *cloud);
    pcl::PointCloud<PointXYZRGB>::Ptr filtered(new pcl::PointCloud<PointXYZRGB>());
    PassThrough<PointXYZRGB> filter;
    filter.setInputCloud(cloud);
    filter.filter(*filtered);
    cout << "passthrough filter complete" << endl;

    cout << "begin normal estimation" << endl;
    NormalEstimationOMP<PointXYZRGB, Normal> ne;//计算点云法向
    ne.setNumberOfThreads(8);//设定临近点
    ne.setInputCloud(filtered);
    ne.setRadiusSearch(0.01);//设定搜索半径
    Eigen::Vector4f centroid;

    compute3DCentroid(*filtered, centroid);//计算点云中心
    ne.setViewPoint(centroid[0], centroid[1], centroid[2]);//将向量计算原点置于点云中心

    pcl::PointCloud<Normal>::Ptr cloud_normals (new pcl::PointCloud<Normal>());
    ne.compute(*cloud_normals);
    cout << "normal estimation complete" << endl;
    cout << "reverse normals' direction" << endl;

//将法向量反向
    for(size_t i = 0; i < cloud_normals->size(); ++i)
    {
        cloud_normals->points[i].normal_x *= -1;
        cloud_normals->points[i].normal_y *= -1;
        cloud_normals->points[i].normal_z *= -1;
    }

//融合RGB点云和法向
    cout << "combine points and normals" << endl;
    pcl::PointCloud<PointXYZRGBNormal>::Ptr cloud_smoothed_normals(new pcl::PointCloud<PointXYZRGBNormal>());
    concatenateFields(*filtered, *cloud_normals, *cloud_smoothed_normals);

//泊松重建
    cout << "begin poisson reconstruction" << endl;
    Poisson<PointXYZRGBNormal> poisson;
    //poisson.setDegree(2);
    poisson.setDepth(8);
    poisson.setSolverDivide (6);
    poisson.setIsoDivide (6);

    poisson.setConfidence(false);
    poisson.setManifold(false);
    poisson.setOutputPolygons(false);

    poisson.setInputCloud(cloud_smoothed_normals);
    PolygonMesh mesh;
    poisson.reconstruct(mesh);

    cout << "finish poisson reconstruction" << endl;

//给mesh染色
    pcl::PointCloud<PointXYZRGB> cloud_color_mesh;
    pcl::fromPCLPointCloud2(mesh.cloud, cloud_color_mesh);

    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud (cloud);
    // K nearest neighbor search
    int K = 5;
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    for(int i=0;i<cloud_color_mesh.points.size();++i)
    {
        uint8_t r = 0;
        uint8_t g = 0;
        uint8_t b = 0;
        float dist = 0.0;
        int red = 0;
        int green = 0;
        int blue = 0;
        uint32_t rgb;

        if ( kdtree.nearestKSearch (cloud_color_mesh.points[i], K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
        {
            for (int j = 0; j < pointIdxNKNSearch.size (); ++j)
            {

                r = cloud->points[ pointIdxNKNSearch[j] ].r;
                g = cloud->points[ pointIdxNKNSearch[j] ].g;
                b = cloud->points[ pointIdxNKNSearch[j] ].b;

                red += int(r);
                green += int(g);
                blue += int(b);
                dist += 1.0/pointNKNSquaredDistance[j];

                //std::cout<<"red: "<<int(r)<<std::endl;
                //std::cout<<"green: "<<int(g)<<std::endl;
                //std::cout<<"blue: "<<int(b)<<std::endl;
                //cout<<"dis:"<<dist<<endl;
            }
        }
        cloud_color_mesh.points[i].r = int(red/pointIdxNKNSearch.size ()+0.5);
        cloud_color_mesh.points[i].g = int(green/pointIdxNKNSearch.size ()+0.5);
        cloud_color_mesh.points[i].b = int(blue/pointIdxNKNSearch.size ()+0.5);
    }
    pcl::toPCLPointCloud2(cloud_color_mesh, mesh.cloud);
    io::savePLYFile("object_mesh.ply", mesh);
    pcl::PCDWriter writer;
    writer.write<pcl::PointXYZRGB>("object_mesh.pcd", cloud_color_mesh);
}


void PointCloudMapping::cpf_seg(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr input_cloud_ptr)
{
    APC::Segmentation seg;
    APC::Config config;
    config.noise_threshold = 0.01; // 1cm
    //config.voxel_resolution = 0.005f;// 0.5cm
    //config.seed_resolution = 0.05f; // 5cm
    //config.min_plane_area = 0.01f; // m^2;
    config.min_plane_area = 0.05f;
    config.max_curvature = 0.01;
    seg.setConfig(config);
    pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
    pcl::VoxelGrid<PointT> sor;
    sor.setInputCloud(input_cloud_ptr);
    sor.setLeafSize (0.005f, 0.005f, 0.005f);
    sor.filter(*cloud_filtered);
    std::cerr << "Number of points after filtered " << cloud_filtered->size() << std::endl;
    seg.setPointCloud(input_cloud_ptr);
    seg.doSegmentation();
    pcl::PointCloud<pcl::PointXYZL>::Ptr segmented_cloud_ptr;
    segmented_cloud_ptr = seg.getSegmentedPointCloud();
    bool save_binary_pcd = false;
    pcl::io::savePCDFile ("segmentation.pcd", *segmented_cloud_ptr, save_binary_pcd);
    cout <<"finish segmentation,please see details in segmentation.pcd"<<endl;
}


pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::CEC(const std::string& file_name) {

    pcl::PointCloud<PointTypeIO>::Ptr cloud_in(new pcl::PointCloud<PointTypeIO>), cloud_out(
            new pcl::PointCloud<PointTypeIO>);
    pcl::PointCloud<PointTypeFull>::Ptr cloud_with_normals(new pcl::PointCloud<PointTypeFull>);
    pcl::IndicesClustersPtr clusters(new pcl::IndicesClusters), small_clusters(
            new pcl::IndicesClusters), large_clusters(new pcl::IndicesClusters);
    pcl::search::KdTree<PointTypeIO>::Ptr search_tree(new pcl::search::KdTree<PointTypeIO>);

    pcl::io::loadPCDFile (file_name, *cloud_in);
    pcl::VoxelGrid<PointTypeIO> vg;

    vg.setInputCloud (cloud_in);
    vg.setLeafSize (80.0, 80.0, 80.0);
    vg.setDownsampleAllData (true);
    vg.filter (*cloud_out);
    pcl::copyPointCloud (*cloud_out, *cloud_with_normals);
    pcl::NormalEstimation<PointTypeIO, PointTypeFull> ne;
    ne.setInputCloud (cloud_out);
    ne.setSearchMethod (search_tree);
    ne.setRadiusSearch (300.0);
    ne.compute (*cloud_with_normals);
    pcl::ConditionalEuclideanClustering<PointTypeFull> cec (true);
    cec.setInputCloud (cloud_with_normals);
    //cec.setConditionFunction (&customRegionGrowing);
    cec.setClusterTolerance (500.0);
    cec.setMinClusterSize (cloud_with_normals->points.size () / 1000);
    cec.setMaxClusterSize (cloud_with_normals->points.size () / 5);
    cec.segment (*clusters);
    cec.getRemovedClusters (small_clusters, large_clusters);

    // Using the intensity channel for lazy visualization of the output
    for (int i = 0; i < small_clusters->size (); ++i)
        for (int j = 0; j < (*small_clusters)[i].indices.size (); ++j)
            cloud_out->points[(*small_clusters)[i].indices[j]].intensity = -2.0;
    for (int i = 0; i < large_clusters->size (); ++i)
        for (int j = 0; j < (*large_clusters)[i].indices.size (); ++j)
            cloud_out->points[(*large_clusters)[i].indices[j]].intensity = +10.0;
    for (int i = 0; i < clusters->size (); ++i)
    {
        int label = rand () % 8;
        for (int j = 0; j < (*clusters)[i].indices.size (); ++j)
            cloud_out->points[(*clusters)[i].indices[j]].intensity = label;
    }
}

pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::ECE(PointCloud::Ptr cloud)
{
    //std::time_t start = std::time(nullptr);
    // std::cout << "PointCloud before filtering has: " << cloud->points.size() << " data points." << std::endl;
    pcl::VoxelGrid<PointT> vg;
    PointCloud::Ptr cloud_filtered (new PointCloud);
    PointCloud::Ptr cloud_f (new PointCloud);
    vg.setInputCloud(cloud);
    vg.setLeafSize(0.01f, 0.01f, 0.01f);
    vg.filter(* cloud_filtered);
    // std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size() << " data points." << std::endl;
    // std::cout << "Wall time passed: "
    //           << std::difftime(std::time(nullptr), start) << " s.\n";

    pcl::SACSegmentation<PointT> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    PointCloud::Ptr cloud_plane (new PointCloud ());
    pcl::PCDWriter writer;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.02);

    int nr_points = (int) cloud_filtered->points.size();
    while(cloud_filtered->points.size() > 0.3 * nr_points)
    {
        seg.setInputCloud(cloud_filtered);
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size() ==0)
        {
            std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
            break;
        }

        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud (cloud_filtered);
        extract.setIndices(inliers);
        extract.setNegative(false);

        extract.filter(*cloud_plane);
        std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

        extract.setNegative(true);
        extract.filter (*cloud_f);
        *cloud_filtered = *cloud_f;
    }

    // Creating the kdTree object for the search method of the extraction
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    tree->setInputCloud(cloud_filtered);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance(0.02); //2cm
    ec.setMinClusterSize(100);
    ec.setMaxClusterSize(25000);
    ec.setInputCloud(cloud_filtered);
    ec.extract(cluster_indices);
    PointCloud::Ptr cloud_cluster (new PointCloud);

    //int j = 0;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
    {

        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end();++pit)
            cloud_cluster->points.push_back (cloud_filtered->points[*pit]);
        cloud_cluster->width = cloud_cluster->points.size();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // std::cout << "PointCloud representing the Cluster: "<< cloud_cluster->points.size() << "data points." << std::endl;
        // std::stringstream ss;
        // ss << "cloud_cluster_"<< j << ".pcd";
        // writer.write<PointT> (ss.str(),*cloud_cluster,false);
        // j++;
    }
    return cloud_cluster;
}

pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::cylinderSeg(PointCloud::Ptr cloud)
{
    pcl::PassThrough<PointT> pass;
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
    pcl::ExtractIndices<PointT> extract;
    pcl::ExtractIndices<pcl::Normal> extract_normals;
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);

    pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<PointT>::Ptr cloud_filtered2 (new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
    pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients), coefficients_cylinder (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices), inliers_cylinder (new pcl::PointIndices);

  // Build a passthrough filter to remove spurious NaNs
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0, 1.5);
    pass.filter (*cloud_filtered);
    std::cerr << "PointCloud after filtering has: " << cloud_filtered->points.size () << " data points." << std::endl;

  // Build a passthrough filter to remove spurious NaNs
    ne.setSearchMethod (tree);
    ne.setInputCloud (cloud_filtered);
    ne.setKSearch (50);
    ne.compute (*cloud_normals);
  // Create the segmentation object for the planar model and set all the parameters

    seg.setOptimizeCoefficients(true);
    seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight(0.1);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.03);
    seg.setInputCloud(cloud_filtered);
    seg.setInputNormals(cloud_normals);
    // Obtain the plane inliers and coefficients
    seg.segment (*inliers_plane, *coefficients_plane);
    std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

    // Extract the planar inliers from the input cloud
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers_plane);
    extract.setNegative (false);
    pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT> ());
    extract.filter (*cloud_plane);

     //Remove the planar inliers, extract the rest
//     extract.setNegative (true);
//     extract.filter (*cloud_filtered2);
//     extract_normals.setNegative (true);
//     extract_normals.setInputCloud (cloud_normals);
//     extract_normals.setIndices (inliers_plane);
//     extract_normals.filter (*cloud_normals2);
//
//     // Create the segmentation object for cylinder segmentation and set all the parameters
//     seg.setOptimizeCoefficients (true);
//     seg.setModelType (pcl::SACMODEL_CYLINDER);
//     seg.setMethodType (pcl::SAC_RANSAC);
//     seg.setNormalDistanceWeight (0.1);
//     seg.setMaxIterations (10000);
//     seg.setDistanceThreshold (0.05);
//     seg.setRadiusLimits (0, 0.1);
//     seg.setInputCloud (cloud_filtered2);
//     seg.setInputNormals (cloud_normals2);
//
//     // Obtain the cylinder inliers and coefficients
//     seg.segment (*inliers_cylinder, *coefficients_cylinder);
//     std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;

//     // Write the cylinder inliers to disk
//     extract.setInputCloud (cloud_filtered2);
//     extract.setIndices (inliers_cylinder);
//     extract.setNegative (false);
//     pcl::PointCloud<PointT>::Ptr cloud_cylinder (new pcl::PointCloud<PointT> ());
//     extract.filter (*cloud_cylinder);

    return cloud_plane;
}


pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::regionGrowingSeg(PointCloud::Ptr cloud_in)
{

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  PointCloudXYZRGBAtoXYZ(*cloud_in,*cloud);
  pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normals (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (cloud);
  normal_estimator.setKSearch (50);
  normal_estimator.compute (*normals);

  pcl::IndicesPtr indices (new std::vector <int>);
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 1.0);
  pass.filter (*indices);

  pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize (50);
  reg.setMaxClusterSize (1000000);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (30);
  reg.setInputCloud (cloud);
  //reg.setIndices (indices);
  reg.setInputNormals (normals);
  reg.setSmoothnessThreshold (3.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold (1.0);

  std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters);
//
//  std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;
//  std::cout << "First cluster has " << clusters[0].indices.size () << " points." << endl;
//  std::cout << "These are the indices of the points of the initial" <<
//    std::endl << "cloud that belong to the first cluster:" << std::endl;
  int counter = 0;
  while (counter < clusters[0].indices.size ())
  {
    std::cout << clusters[0].indices[counter] << ", ";
    counter++;
    if (counter % 10 == 0)
      std::cout << std::endl;
  }

  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
//  pcl::visualization::CloudViewer viewer ("Cluster viewer");
//  viewer.showCloud(colored_cloud);
//  while (!viewer.wasStopped ())
//  {
//  }

  pcl::PointCloud <pcl::PointXYZRGBA>::Ptr cloud_out (new pcl::PointCloud <pcl::PointXYZRGBA>);
  PointCloudXYZRGBtoXYZRGBA(*colored_cloud, *cloud_out);
  return cloud_out ;
}


pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::colorRegionGrowingSeg(pcl::PointCloud <pcl::PointXYZRGB>::Ptr  cloud)
{

    pcl::search::Search <pcl::PointXYZRGB>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZRGB> > (new pcl::search::KdTree<pcl::PointXYZRGB>);;

    pcl::IndicesPtr indices (new std::vector <int>);
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.0, 1.0);
    pass.filter (*indices);

    pcl::RegionGrowingRGB<pcl::PointXYZRGB> reg;
    reg.setInputCloud (cloud);
    reg.setIndices (indices);
    reg.setSearchMethod (tree);
    reg.setDistanceThreshold (10);
    reg.setPointColorThreshold (6);
    reg.setRegionColorThreshold (5);
    reg.setMinClusterSize (600);

    std::vector <pcl::PointIndices> clusters;
    reg.extract (clusters);

    pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();


    pcl::PointCloud <pcl::PointXYZRGBA>::Ptr cloud_out (new pcl::PointCloud <pcl::PointXYZRGBA>);
    PointCloudXYZRGBtoXYZRGBA(*colored_cloud, *cloud_out);
    return cloud_out;
}



bool
enforceIntensitySimilarity (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
    if (fabs (point_a.intensity - point_b.intensity) < 5.0f)
        return (true);
    else
        return (false);
}

bool
enforceCurvatureOrIntensitySimilarity (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
    Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
    if (fabs (point_a.intensity - point_b.intensity) < 5.0f)
        return (true);
    if (fabs (point_a_normal.dot (point_b_normal)) < 0.05)
        return (true);
    return (false);
}

bool customRegionGrowing (const PointTypeFull& point_a, const PointTypeFull& point_b, float squared_distance)
{
    Eigen::Map<const Eigen::Vector3f> point_a_normal = point_a.getNormalVector3fMap (), point_b_normal = point_b.getNormalVector3fMap ();
    if (squared_distance < 10000)
    {
        if (fabs (point_a.intensity - point_b.intensity) < 8.0f)
            return (true);
        if (fabs (point_a_normal.dot (point_b_normal)) < 0.06)
            return (true);
    }
    else
    {
        if (fabs (point_a.intensity - point_b.intensity) < 3.0f)
            return (true);
    }
    return (false);
}
