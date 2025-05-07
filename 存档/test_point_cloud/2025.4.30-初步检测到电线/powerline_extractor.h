#ifndef POWERLINE_EXTRACTOR_H
#define POWERLINE_EXTRACTOR_H

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <matio.h>
#include <string>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/segmentation/sac_segmentation.h>  // 新增
#include <pcl/filters/extract_indices.h>        // 新增


#include <Eigen/Dense>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>

#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <unordered_map>                         // 新增
#include <unordered_set>                         // 新增
#include <omp.h>                                 // 新增，用于并行计算

class PowerlineExtractor {
public:
    PowerlineExtractor();
    ~PowerlineExtractor();
    void processAndPublishPowerlines();

private:
    // 回调函数
    void lidarCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);
    
    // 坐标转换
    bool transformPointCloud(const sensor_msgs::PointCloud2& input_cloud, 
                           sensor_msgs::PointCloud2& output_cloud,
                           const std::string& target_frame);
    
    // 加载.mat文件和基本处理
    void checkParameters();
    bool loadMatFile(const std::string& file_path);
    void doubleToPointCloud();
    void lidarMsgToPointCloud(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);
    
    // 新增方法: 裁减点云
    void clipPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input, 
                      pcl::PointCloud<pcl::PointXYZI>::Ptr& output, 
                      float radius);
    
    // 新增方法: 累积点云
    void accumulatePointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& new_cloud);
    
    //提取非地面点云--主函数
    void extractNonGroundPoints();

    // 新增的地面过滤方法
    void extractNonGroundPointsSimple();     // 简单阈值法
    void extractNonGroundPointsRANSAC();     // RANSAC平面拟合法
    void extractNonGroundPointsAdaptive();   // 自适应网格法
    void extractNonGroundPointsParallel();   // 并行优化方法
    void extractNonGroundPointsOptimized();  // 电力线优化方法


    // 电力线提取相关方法
    void extractPowerlinePoints();
    void getPCA(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, 
               pcl::PointCloud<pcl::PointXYZI>::Ptr& powerlinePoints);
    void clusterPowerlines(const pcl::PointCloud<pcl::PointXYZI>::Ptr& candidateCloud,
                pcl::PointCloud<pcl::PointXYZI>::Ptr& filteredCloud);


    // 新增：后处理电力线点云
    void postProcessPowerlinePoints(pcl::PointCloud<pcl::PointXYZI>::Ptr& powerlineCloud);
    
    // 其他工具方法
    void centerPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);
    void downsamplePointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input, 
                             pcl::PointCloud<pcl::PointXYZI>::Ptr& output);
    void adjustPointCloudOrigin(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

    // ROS相关
    ros::NodeHandle nh_;
    ros::Publisher original_cloud_pub_;
    ros::Publisher clipped_cloud_pub_;  // 新增裁减点云发布器
    ros::Publisher accumulated_cloud_pub_;  // 新增累积点云发布器
    ros::Publisher downsamole_cloud_pub_;
    ros::Publisher non_ground_pub_;
    ros::Publisher powerline_pub_;
    ros::Publisher clustered_powerline_pub_;
    ros::Subscriber lidar_sub_;
    
    // TF相关
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::string target_frame_;
    
    // 数据源选择
    bool use_lidar_data_;
    bool new_lidar_data_available_;
    
    // 参数
    double scale_factor_;
    std::string mat_file_path_;
    double voxel_size_;
    double pca_radius_;
    double angle_threshold_;
    double linearity_threshold_;
    double cluster_tolerance_;
    int min_cluster_size_;
    int max_cluster_size_;
    double clip_radius_;  // 新增点云裁减半径参数
    int max_accumulated_frames_;  // 新增累积帧数参数


    // 新增地面过滤相关参数
    double ground_height_threshold_;      // 地面高度阈值
    std::string ground_filter_method_;    // 地面过滤方法
    bool use_cached_ground_info_;         // 是否使用缓存的地面信息
    float cached_ground_threshold_;       // 缓存的地面高度阈值
    ros::Time last_ground_update_;        // 上次更新地面信息的时间
    
    // .mat文件数据
    double** point_cloud_data_;
    size_t num_points_;
    
    // 点云对象
    pcl::PointCloud<pcl::PointXYZI>::Ptr original_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr clipped_cloud_;  // 新增裁减后的点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr accumulated_cloud_;  // 新增累积点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled_cloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr non_ground_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr powerline_cloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr clustered_powerline_cloud_;
    
    // 滤波器
    pcl::VoxelGrid<pcl::PointXYZI> down_size_filter_;
    
    // 累积点云计数器
    int accumulated_frame_count_;  // 新增累积帧计数器
};

#endif // POWERLINE_EXTRACTOR_H