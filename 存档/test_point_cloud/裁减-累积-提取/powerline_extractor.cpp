#include "powerline_extractor.h"

PowerlineExtractor::PowerlineExtractor() 
    : nh_("~"), 
      tf_listener_(tf_buffer_),
      point_cloud_data_(nullptr), 
      num_points_(0),
      use_lidar_data_(false),
      new_lidar_data_available_(false),
      accumulated_frame_count_(0),
      original_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      clipped_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      accumulated_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      downsampled_cloud(new pcl::PointCloud<pcl::PointXYZI>()),
      non_ground_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      powerline_cloud_(new pcl::PointCloud<pcl::PointXYZI>()),
      clustered_powerline_cloud_(new pcl::PointCloud<pcl::PointXYZI>()) {
    
    // 读取参数
    nh_.param<double>("scale_factor", scale_factor_, 1.0);
    nh_.param<std::string>("data_folder", mat_file_path_, "");
    nh_.param<double>("voxel_size", voxel_size_, 0.05);
    nh_.param<double>("pca_radius", pca_radius_, 0.5);
    nh_.param<double>("angle_threshold", angle_threshold_, 10.0);
    nh_.param<double>("linearity_threshold", linearity_threshold_, 0.98);
    nh_.param<double>("cluster_tolerance", cluster_tolerance_, 2.0);
    nh_.param<int>("min_cluster_size", min_cluster_size_, 15);
    nh_.param<int>("max_cluster_size", max_cluster_size_, 100000);
    nh_.param<bool>("use_lidar_data", use_lidar_data_, false);
    nh_.param<std::string>("target_frame", target_frame_, "map");
    // 新增参数
    nh_.param<double>("clip_radius", clip_radius_, 8.0);
    nh_.param<int>("max_accumulated_frames", max_accumulated_frames_, 10);
    
    // 检查参数
    checkParameters();
    
    // 初始化发布器
    original_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("original_cloud", 1);
    clipped_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("clipped_cloud", 1); // 新增
    accumulated_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("accumulated_cloud", 1); // 新增
    downsamole_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("downsamole_cloud", 1);
    non_ground_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("non_ground_cloud", 1);
    powerline_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("powerline_cloud", 1);
    clustered_powerline_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("clustered_powerline_cloud", 1);
    
    // 如果使用LiDAR数据，设置订阅器
    if (use_lidar_data_) {
        lidar_sub_ = nh_.subscribe("/rslidar_points", 1, &PowerlineExtractor::lidarCallback, this);
        ROS_INFO("Subscribed to /rslidar_points topic for LiDAR data");
    } else {
        // 加载.mat文件
        if (!loadMatFile(mat_file_path_)) {
            ROS_ERROR("Failed to load .mat file: %s, shutting down", mat_file_path_.c_str());
            ros::shutdown();
        }
    }
}

PowerlineExtractor::~PowerlineExtractor() {
    // 释放内存
    if (point_cloud_data_) {
        for (size_t i = 0; i < num_points_; ++i) {
            delete[] point_cloud_data_[i];
        }
        delete[] point_cloud_data_;
    }
}


void PowerlineExtractor::checkParameters() {
    ROS_INFO("Parameter 'scale_factor': %f", scale_factor_);
    ROS_INFO("Parameter 'data_folder': %s", mat_file_path_.c_str());
    ROS_INFO("Parameter 'voxel_size': %f", voxel_size_);
    ROS_INFO("Parameter 'pca_radius': %f", pca_radius_);
    ROS_INFO("Parameter 'angle_threshold': %f", angle_threshold_);
    ROS_INFO("Parameter 'linearity_threshold': %f", linearity_threshold_);
    ROS_INFO("Parameter 'cluster_tolerance': %f", cluster_tolerance_);
    ROS_INFO("Parameter 'min_cluster_size': %d", min_cluster_size_);
    ROS_INFO("Parameter 'max_cluster_size': %d", max_cluster_size_);
    ROS_INFO("Parameter 'use_lidar_data': %d", use_lidar_data_);
    ROS_INFO("Parameter 'target_frame': %s", target_frame_.c_str());
    // 新增参数日志
    ROS_INFO("Parameter 'clip_radius': %f", clip_radius_);
    ROS_INFO("Parameter 'max_accumulated_frames': %d", max_accumulated_frames_);
    
    if (scale_factor_ <= 0) {
        ROS_WARN("scale_factor is invalid (<= 0), using default value 1.0");
        scale_factor_ = 1.0;
    }
    
    if (!use_lidar_data_ && mat_file_path_.empty()) {
        ROS_ERROR("data_folder parameter is not set or empty and not using LiDAR data");
        ros::shutdown();
    }
    
    // 检查新增参数
    if (clip_radius_ <= 0) {
        ROS_WARN("clip_radius is invalid (<= 0), using default value 8.0");
        clip_radius_ = 8.0;
    }
    
    if (max_accumulated_frames_ <= 0) {
        ROS_WARN("max_accumulated_frames is invalid (<= 0), using default value 10");
        max_accumulated_frames_ = 10;
    }
}

void PowerlineExtractor::lidarCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    ROS_INFO("Received LiDAR point cloud with %d points, frame_id: %s", 
             cloud_msg->width * cloud_msg->height, cloud_msg->header.frame_id.c_str());
    
    // 转换点云坐标系到目标坐标系
    sensor_msgs::PointCloud2 transformed_cloud;
    if (!transformPointCloud(*cloud_msg, transformed_cloud, target_frame_)) {
        ROS_WARN("Failed to transform point cloud from %s to %s", 
                 cloud_msg->header.frame_id.c_str(), target_frame_.c_str());
        return;
    }
    
    // 将LiDAR消息转换为PCL点云
    lidarMsgToPointCloud(boost::make_shared<sensor_msgs::PointCloud2>(transformed_cloud));
    
    // 标记有新数据可用
    new_lidar_data_available_ = true;
}

bool PowerlineExtractor::transformPointCloud(const sensor_msgs::PointCloud2& input_cloud, 
                                           sensor_msgs::PointCloud2& output_cloud,
                                           const std::string& target_frame) {
    if (input_cloud.header.frame_id == target_frame) {
        output_cloud = input_cloud;
        return true;
    }
    
    try {
        // 等待坐标变换可用
        if (tf_buffer_.canTransform(target_frame, input_cloud.header.frame_id, input_cloud.header.stamp, ros::Duration(1.0))) {
            // 执行坐标变换
            tf_buffer_.transform(input_cloud, output_cloud, target_frame);
            return true;
        } else {
            ROS_WARN("Transform from %s to %s not available", 
                     input_cloud.header.frame_id.c_str(), target_frame.c_str());
            return false;
        }
    } catch (tf2::TransformException& ex) {
        ROS_WARN("Transform exception: %s", ex.what());
        return false;
    }
}

void PowerlineExtractor::lidarMsgToPointCloud(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    // 清空现有点云
    original_cloud_->clear();
    
    // 使用PCL转换器从ROS消息转换到PCL点云格式
    pcl::fromROSMsg(*cloud_msg, *original_cloud_);
    
    ROS_INFO("Converted LiDAR message to PCL point cloud with %zu points", original_cloud_->size());
    
    // 记录原始点云边界
    pcl::PointXYZI min_point, max_point;
    pcl::getMinMax3D(*original_cloud_, min_point, max_point);
    ROS_INFO("LiDAR point cloud bounds: X[%.2f, %.2f], Y[%.2f, %.2f], Z[%.2f, %.2f]", 
             min_point.x, max_point.x, min_point.y, max_point.y, min_point.z, max_point.z);
}

bool PowerlineExtractor::loadMatFile(const std::string& file_path) {
    // 打开.mat文件
    mat_t* matfp = Mat_Open(file_path.c_str(), MAT_ACC_RDONLY);
    if (!matfp) {
        ROS_ERROR("Cannot open .mat file: %s", file_path.c_str());
        return false;
    }
    
    // 读取ptCloudA结构体
    matvar_t* matvar = Mat_VarRead(matfp, "ptCloudA");
    if (!matvar || matvar->class_type != MAT_C_STRUCT) {
        ROS_ERROR("Failed to read ptCloudA struct");
        Mat_Close(matfp);
        return false;
    }
    
    // 获取data字段
    matvar_t* data_field = Mat_VarGetStructFieldByName(matvar, "data", 0);
    if (!data_field || data_field->class_type != MAT_C_DOUBLE) {
        ROS_ERROR("Failed to read ptCloudA.data field");
        Mat_VarFree(matvar);
        Mat_Close(matfp);
        return false;
    }
    
    // 获取维度
    size_t* dims = data_field->dims;
    num_points_ = dims[0];
    size_t num_fields = dims[1];
    if (num_fields < 3) {
        ROS_ERROR("Data has fewer than 3 columns (x, y, z)");
        Mat_VarFree(matvar);
        Mat_Close(matfp);
        return false;
    }
    ROS_INFO("Loaded point cloud with dimensions: %zu x %zu", num_points_, num_fields);
    
    // 分配内存
    point_cloud_data_ = new double*[num_points_];
    for (size_t i = 0; i < num_points_; ++i) {
        point_cloud_data_[i] = new double[3]; // 存储x, y, z
    }
    
    // 复制x, y, z数据
    double* data_ptr = static_cast<double*>(data_field->data);
    for (size_t i = 0; i < num_points_; ++i) {
        point_cloud_data_[i][0] = data_ptr[0 * num_points_ + i] / scale_factor_; // x
        point_cloud_data_[i][1] = data_ptr[1 * num_points_ + i] / scale_factor_; // y
        point_cloud_data_[i][2] = data_ptr[2 * num_points_ + i]; // z
    }
    
    ROS_INFO("Loaded %zu points from .mat file", num_points_);
    
    // 清理
    Mat_VarFree(matvar);
    Mat_Close(matfp);
    return true;
}

void PowerlineExtractor::doubleToPointCloud() {
    original_cloud_->clear();
    pcl::PointXYZI point;
    
    for (size_t i = 0; i < num_points_; ++i) {
        point.x = point_cloud_data_[i][0] -320700+50 ;
        point.y = point_cloud_data_[i][1] -4783000-100 ;
        point.z = point_cloud_data_[i][2] -260 ;
        point.intensity = 1.0;
        original_cloud_->push_back(point);
    }
    
    ROS_INFO("Converted %zu points to PCL point cloud", original_cloud_->size());
}

void PowerlineExtractor::adjustPointCloudOrigin(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    // 找到点云的边界
    pcl::PointXYZI min_point, max_point;
    pcl::getMinMax3D(*cloud, min_point, max_point);
    
    // 计算中心点和最小Z值
    float center_x = (min_point.x + max_point.x) / 2.0f;
    float center_y = (min_point.y + max_point.y) / 2.0f;
    float min_z = min_point.z;
    
    ROS_INFO("Adjusting point cloud origin: Center(%.2f, %.2f), Min Z: %.2f", 
             center_x, center_y, min_z);
    
    // 将点云平移到中心
    for (auto& pt : cloud->points) {
        pt.x -= center_x;
        pt.y -= center_y;
        pt.z -= min_z;
    }
}



void PowerlineExtractor::extractNonGroundPoints() {
    non_ground_cloud_->clear();
    
    // 使用直方图方法找到地面高度
    const int num_bins = 50;
    std::vector<int> histogram(num_bins, 0);
    
    // 找到z值的范围
    float min_z = std::numeric_limits<float>::max();
    float max_z = std::numeric_limits<float>::lowest();
    
    for (const auto& point : downsampled_cloud->points) {
        min_z = std::min(min_z, point.z);
        max_z = std::max(max_z, point.z);
    }
    
    // 检查点云的高度范围
    ROS_INFO("Z value range: [%.2f, %.2f]", min_z, max_z);
    
    // 防止除以零或值太小导致的问题
    float z_range = max_z - min_z;
    if (z_range < 0.01) { // 设置一个最小差值阈值
        ROS_WARN("Z value range too small (%.6f), setting all points as non-ground", z_range);
        *non_ground_cloud_ = *downsampled_cloud;
        return;
    }
    
    float bin_size = z_range / num_bins;
    ROS_INFO("Bin size for histogram: %.4f", bin_size);
    
    // 构建高度直方图
    for (const auto& point : downsampled_cloud->points) {
        int bin_idx = std::min(static_cast<int>((point.z - min_z) / bin_size), num_bins - 1);
        if (bin_idx >= 0 && bin_idx < num_bins) { // 确保索引在有效范围内
            histogram[bin_idx]++;
        }
    }
    
    // 输出直方图信息，便于调试
    ROS_INFO("Height histogram:");
    for (int i = 0; i < num_bins; i++) {
        if (histogram[i] > 0) {
            ROS_INFO("  Bin %d: %d points", i, histogram[i]);
        }
    }
    
    // 找到直方图中的最大值
    int max_count = 0;
    int max_bin_idx = 0;
    
    for (int i = 0; i < num_bins; i++) {
        if (histogram[i] > max_count) {
            max_count = histogram[i];
            max_bin_idx = i;
        }
    }
    
    ROS_INFO("Maximum bin: %d with %d points", max_bin_idx, max_count);
    
    // 在最大值上方3个bin设置地面阈值
    float ground_threshold = min_z + (max_bin_idx + 3) * bin_size;
    ROS_INFO("Ground threshold set at: %.2f", ground_threshold);
    
    // 筛选非地面点
    for (const auto& point : downsampled_cloud->points) {
        if (point.z > ground_threshold) {
            non_ground_cloud_->push_back(point);
        }
    }
    
    ROS_INFO("Extracted %zu non-ground points", non_ground_cloud_->size());
}

// void PowerlineExtractor::downsamplePointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input, 
//                                              pcl::PointCloud<pcl::PointXYZI>::Ptr& output) {
//     pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
//     voxel_filter.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
//     voxel_filter.setInputCloud(input);
//     voxel_filter.filter(*output);
    
//     ROS_INFO("Downsampled point cloud from %zu to %zu points", 
//              input->size(), output->size());
// }

void PowerlineExtractor::downsamplePointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input, 
    pcl::PointCloud<pcl::PointXYZI>::Ptr& output) {
// 检查点云是否为空
if (input->empty()) {
ROS_WARN("Input cloud is empty, skipping downsampling");
output->clear();
return;
}

// 检查叶子大小
if (voxel_size_ < 0.01) {
ROS_WARN("Voxel size too small (%.4f), increasing to 0.1", voxel_size_);
voxel_size_ = 0.1; // 设置一个更合理的默认值
}

// 创建一个临时点云存储有效点
pcl::PointCloud<pcl::PointXYZI>::Ptr valid_cloud(new pcl::PointCloud<pcl::PointXYZI>());

// 过滤无效点
for (const auto& point : input->points) {
if (pcl::isFinite(point)) {
valid_cloud->push_back(point);
}
}

ROS_INFO("Removed %zu invalid points", input->size() - valid_cloud->size());

// 如果所有点都无效，返回空点云
if (valid_cloud->empty()) {
ROS_WARN("No valid points after filtering, returning empty cloud");
output->clear();
return;
}

// 执行下采样
pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
voxel_filter.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
voxel_filter.setInputCloud(valid_cloud);
voxel_filter.filter(*output);

ROS_INFO("Downsampled point cloud from %zu to %zu points", 
valid_cloud->size(), output->size());
}

void PowerlineExtractor::centerPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
    pcl::PointXYZI min_point, max_point;
    pcl::getMinMax3D(*cloud, min_point, max_point);
    
    Eigen::Vector4f centroid;
    centroid[0] = (min_point.x + max_point.x) / 2;
    centroid[1] = (min_point.y + max_point.y) / 2;
    centroid[2] = (min_point.z + max_point.z) / 2;
    centroid[3] = 1.0f;
    
    for (auto& point : cloud->points) {
        point.x -= centroid[0];
        point.y -= centroid[1];
        point.z -= centroid[2];
    }
    
    ROS_INFO("Centered point cloud around (%.2f, %.2f, %.2f)", 
             centroid[0], centroid[1], centroid[2]);
}

// void PowerlineExtractor::getPCA(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, 
//                               pcl::PointCloud<pcl::PointXYZI>::Ptr& powerlinePoints) {
//     powerlinePoints->clear();
    
//     // 创建KD树用于近邻搜索
//     pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
//     kdtree.setInputCloud(cloud);
    
//     std::vector<int> pointIdxRadiusSearch;
//     std::vector<float> pointRadiusSquaredDistance;
    
//     // 对每个点计算PCA
//     for (size_t i = 0; i < cloud->size(); ++i) {
//         // 搜索半径内的近邻点
//         if (kdtree.radiusSearch(cloud->points[i], pca_radius_, 
//                                pointIdxRadiusSearch, pointRadiusSquaredDistance) < 3) {
//             continue; // 需要至少3个点进行PCA
//         }
        
//         // 将近邻点收集到一个矩阵中
//         Eigen::MatrixXf neighborhood(pointIdxRadiusSearch.size(), 3);
//         for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j) {
//             neighborhood(j, 0) = cloud->points[pointIdxRadiusSearch[j]].x;
//             neighborhood(j, 1) = cloud->points[pointIdxRadiusSearch[j]].y;
//             neighborhood(j, 2) = cloud->points[pointIdxRadiusSearch[j]].z;
//         }
        
//         // 计算近邻点的协方差矩阵
//         Eigen::MatrixXf centered = neighborhood.rowwise() - neighborhood.colwise().mean();
//         Eigen::MatrixXf cov = (centered.transpose() * centered) / float(neighborhood.rows() - 1);
        
//         // 计算特征值和特征向量
//         Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(cov);
//         Eigen::Vector3f eigenvalues = eig.eigenvalues();
//         Eigen::Matrix3f eigenvectors = eig.eigenvectors();
        
//         // 确保特征值是降序排列的（由大到小）
//         std::vector<std::pair<float, int>> eigenvalue_indices;
//         for (int j = 0; j < 3; ++j) {
//             eigenvalue_indices.push_back(std::make_pair(eigenvalues(j), j));
//         }
//         std::sort(eigenvalue_indices.begin(), eigenvalue_indices.end(), 
//                  [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
//                      return a.first > b.first;
//                  });
        
//         // 获取排序后的特征值和特征向量
//         float lambda1 = eigenvalues(eigenvalue_indices[0].second);
//         float lambda2 = eigenvalues(eigenvalue_indices[1].second);
//         float lambda3 = eigenvalues(eigenvalue_indices[2].second);
//         Eigen::Vector3f normal = eigenvectors.col(eigenvalue_indices[0].second);
        
//         // 计算线性度
//         float linearity = (lambda1 - lambda2) / lambda1;
        
//         // 计算法向量与垂直轴的夹角（度）
//         float angle = std::acos(std::abs(normal(2)) / normal.norm()) * 180.0 / M_PI;
        
//         // 检查是否是电力线的条件：接近水平且高度线性
//         if (std::abs(angle - 90.0) < angle_threshold_ && linearity > linearity_threshold_) {
//             powerlinePoints->push_back(cloud->points[i]);
//         }
//     }
    
//     ROS_INFO("Extracted %zu powerline points using PCA", powerlinePoints->size());
// }

void PowerlineExtractor::getPCA(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, 
    pcl::PointCloud<pcl::PointXYZI>::Ptr& powerlinePoints) {
powerlinePoints->clear();

if (cloud->empty()) {
ROS_WARN("Input cloud for PCA is empty");
return;
}

// 检查所有点是否有效
bool has_invalid_points = false;
for (const auto& point : cloud->points) {
if (!pcl::isFinite(point)) {
has_invalid_points = true;
break;
}
}

if (has_invalid_points) {
ROS_WARN("Cloud contains invalid points, filtering them out");
pcl::PointCloud<pcl::PointXYZI>::Ptr valid_cloud(new pcl::PointCloud<pcl::PointXYZI>());

for (const auto& point : cloud->points) {
if (pcl::isFinite(point)) {
valid_cloud->push_back(point);
}
}

if (valid_cloud->empty()) {
ROS_WARN("No valid points after filtering");
return;
}

// 递归调用自身，使用有效点
getPCA(valid_cloud, powerlinePoints);
return;
}

// 创建KD树用于近邻搜索
pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
kdtree.setInputCloud(cloud);

std::vector<int> pointIdxRadiusSearch;
std::vector<float> pointRadiusSquaredDistance;

// 使用较大的搜索半径
double search_radius = pca_radius_ * 1.5;
// 使用较大的线性阈值
double adjusted_linearity_threshold = linearity_threshold_ * 0.9;
// 使用较宽松的角度阈值
double adjusted_angle_threshold = angle_threshold_ * 1.5;

ROS_INFO("Using adjusted PCA parameters - Search radius: %.2f, Linearity threshold: %.4f, Angle threshold: %.2f",
search_radius, adjusted_linearity_threshold, adjusted_angle_threshold);

// 存储每个点的线性度和角度，用于后期筛选
std::vector<std::tuple<size_t, float, float>> point_features; // 索引、线性度、角度

// 第一轮：计算每个点的特征
for (size_t i = 0; i < cloud->size(); ++i) {
try {
// 检查当前点是否有效
if (!pcl::isFinite(cloud->points[i])) {
continue;
}

// 搜索半径内的近邻点
if (kdtree.radiusSearch(cloud->points[i], search_radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) < 5) {
continue; // 需要足够的点进行更稳定的PCA
}

// 将近邻点收集到一个矩阵中
Eigen::MatrixXf neighborhood(pointIdxRadiusSearch.size(), 3);
for (size_t j = 0; j < pointIdxRadiusSearch.size(); ++j) {
neighborhood(j, 0) = cloud->points[pointIdxRadiusSearch[j]].x;
neighborhood(j, 1) = cloud->points[pointIdxRadiusSearch[j]].y;
neighborhood(j, 2) = cloud->points[pointIdxRadiusSearch[j]].z;
}

// 计算近邻点的协方差矩阵
Eigen::MatrixXf centered = neighborhood.rowwise() - neighborhood.colwise().mean();
Eigen::MatrixXf cov = (centered.transpose() * centered) / float(neighborhood.rows() - 1);

// 计算特征值和特征向量
Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig(cov);
Eigen::Vector3f eigenvalues = eig.eigenvalues();
Eigen::Matrix3f eigenvectors = eig.eigenvectors();

// 确保特征值是降序排列的（由大到小）
std::vector<std::pair<float, int>> eigenvalue_indices;
for (int j = 0; j < 3; ++j) {
eigenvalue_indices.push_back(std::make_pair(eigenvalues(j), j));
}
std::sort(eigenvalue_indices.begin(), eigenvalue_indices.end(), 
 [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
     return a.first > b.first;
 });

// 获取排序后的特征值和特征向量
float lambda1 = eigenvalues(eigenvalue_indices[0].second);
float lambda2 = eigenvalues(eigenvalue_indices[1].second);
float lambda3 = eigenvalues(eigenvalue_indices[2].second);
Eigen::Vector3f normal = eigenvectors.col(eigenvalue_indices[0].second);

// 计算线性度
float linearity = (lambda1 - lambda2) / lambda1;

// 计算法向量与垂直轴的夹角（度）
float angle = std::acos(std::abs(normal(2)) / normal.norm()) * 180.0 / M_PI;

// 存储特征
point_features.push_back(std::make_tuple(i, linearity, angle));
} 
catch (const std::exception& e) {
ROS_WARN("Exception in PCA calculation: %s", e.what());
continue;
}
}

// 根据线性度对特征进行排序 - 首先考虑具有高线性度的点
std::sort(point_features.begin(), point_features.end(),
[](const std::tuple<size_t, float, float>& a, const std::tuple<size_t, float, float>& b) {
return std::get<1>(a) > std::get<1>(b);
});

// 保留线性度最高的点云比例
size_t high_linearity_count = std::min(size_t(cloud->size() * 0.3), point_features.size());  // 保留前30%的高线性度点

// 第二轮：筛选出电力线点 - 从线性度高的点云中优先选择
for (size_t i = 0; i < high_linearity_count; ++i) {
size_t idx = std::get<0>(point_features[i]);
float linearity = std::get<1>(point_features[i]);
float angle = std::get<2>(point_features[i]);

// 第一个条件：线性度符合阈值
bool linearity_ok = linearity > adjusted_linearity_threshold;

// 第二个条件：接近水平
bool angle_ok = std::abs(angle - 90.0) < adjusted_angle_threshold;

// 第三个条件：高度在一定范围内（可选，视情况调整或移除）
// bool height_ok = cloud->points[idx].z > min_power_line_height_;

// 综合条件判断
if (linearity_ok && angle_ok) {
powerlinePoints->push_back(cloud->points[idx]);

// 查找该点附近的其他可能属于同一电力线的点
std::vector<int> neighbors;
std::vector<float> distances;

// 使用较小的半径搜索电力线附近的点
if (kdtree.radiusSearch(cloud->points[idx], pca_radius_ * 0.8, neighbors, distances) > 0) {
for (size_t j = 0; j < neighbors.size(); ++j) {
// 添加邻近点，可能是同一电力线的一部分
powerlinePoints->push_back(cloud->points[neighbors[j]]);
}
}
}
}

// 去除重复点
pcl::PointCloud<pcl::PointXYZI>::Ptr unique_cloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
float small_voxel_size = voxel_size_ * 0.5;  // 使用较小的体素尺寸
voxel_filter.setLeafSize(small_voxel_size, small_voxel_size, small_voxel_size);
voxel_filter.setInputCloud(powerlinePoints);
voxel_filter.filter(*unique_cloud);

// 更新结果
powerlinePoints->swap(*unique_cloud);

ROS_INFO("Extracted %zu powerline points using improved PCA", powerlinePoints->size());
}

// 新增一个后处理电力线点云的方法
void PowerlineExtractor::postProcessPowerlinePoints(pcl::PointCloud<pcl::PointXYZI>::Ptr& powerlineCloud) {
if (powerlineCloud->empty()) {
ROS_WARN("Powerline cloud is empty, skipping post-processing");
return;
}

// 1. 使用半径离群点移除滤波器清除孤立点
pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZI>());
pcl::RadiusOutlierRemoval<pcl::PointXYZI> outrem;
outrem.setInputCloud(powerlineCloud);
outrem.setRadiusSearch(1.0);  // 1米半径内
outrem.setMinNeighborsInRadius(5);  // 至少5个邻居
outrem.filter(*filtered_cloud);

ROS_INFO("Removed %zu outlier points", powerlineCloud->size() - filtered_cloud->size());

// 2. 基于区域生长算法改进点云聚类
powerlineCloud->swap(*filtered_cloud);  // 更新为过滤后的点云
}

void PowerlineExtractor::extractPowerlinePoints() {
    // 对非地面点进行下采样
    // pcl::PointCloud<pcl::PointXYZI>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    // downsamplePointCloud(non_ground_cloud_, downsampled_cloud);
    
    // 使用PCA提取电力线点
    getPCA(non_ground_cloud_, powerline_cloud_);
}

void PowerlineExtractor::clusterPowerlines(const pcl::PointCloud<pcl::PointXYZI>::Ptr& candidateCloud,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& filteredCloud) {
    if (candidateCloud->empty()) {
        ROS_WARN("Candidate powerline cloud is empty, skipping clustering");
        return;
    }

    filteredCloud->clear();

    // 创建KD树用于Euclidean聚类
    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    tree->setInputCloud(candidateCloud);

    // 存储聚类结果
    std::vector<pcl::PointIndices> cluster_indices;

    // 创建欧几里得聚类对象
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(cluster_tolerance_);  // 设置聚类距离阈值
    ec.setMinClusterSize(min_cluster_size_);     // 设置最小聚类点数
    ec.setMaxClusterSize(max_cluster_size_);     // 设置最大聚类点数
    ec.setSearchMethod(tree);
    ec.setInputCloud(candidateCloud);
    ec.extract(cluster_indices);

    ROS_INFO("Found %zu clusters in candidate powerline points", cluster_indices.size());

    // 处理每个聚类
    int cluster_id = 0;
    for (const auto& indices : cluster_indices) {
        cluster_id++;

        // 跳过太小的聚类（冗余检查）
        if (indices.indices.size() < min_cluster_size_) {
            continue;
        }

        // 提取当前聚类的点
        for (const auto& index : indices.indices) {
            pcl::PointXYZI point = candidateCloud->points[index];
            // 可以设置intensity值为聚类ID，便于可视化不同的聚类
            point.intensity = static_cast<float>(cluster_id);
            filteredCloud->push_back(point);
        }
    }

    ROS_INFO("Extracted %zu points from %zu clusters", filteredCloud->size(), cluster_indices.size());
}



// 新增方法：裁减点云
void PowerlineExtractor::clipPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input, 
    pcl::PointCloud<pcl::PointXYZI>::Ptr& output, 
    float radius) {
output->clear();

if (input->empty()) {
ROS_WARN("Input cloud for clipping is empty");
return;
}

// 创建一个临时点云存储有效点
pcl::PointCloud<pcl::PointXYZI>::Ptr valid_cloud(new pcl::PointCloud<pcl::PointXYZI>());

// 首先过滤无效点
for (const auto& point : input->points) {
if (pcl::isFinite(point)) {
valid_cloud->push_back(point);
}
}

if (valid_cloud->empty()) {
ROS_WARN("No valid points after filtering");
return;
}

// 计算点到原点的距离，保留半径内的点
for (const auto& point : valid_cloud->points) {
float distance = std::sqrt(point.x * point.x + point.y * point.y);
if (distance <= radius) {
output->push_back(point);
}
}

ROS_INFO("Clipped point cloud from %zu to %zu points within radius %f", 
valid_cloud->size(), output->size(), radius);
}

// 新增方法：累积点云
void PowerlineExtractor::accumulatePointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& new_cloud) {
if (new_cloud->empty()) {
ROS_WARN("New cloud for accumulation is empty");
return;
}

// 检查累积帧是否达到最大值
if (accumulated_frame_count_ >= max_accumulated_frames_) {
ROS_INFO("Resetting accumulated point cloud (max frames: %d reached)", max_accumulated_frames_);
accumulated_cloud_->clear();
accumulated_frame_count_ = 0;
}

// 将新点云添加到累积点云中
*accumulated_cloud_ += *new_cloud;
accumulated_frame_count_++;

ROS_INFO("Accumulated cloud now has %zu points (frame %d/%d)", 
accumulated_cloud_->size(), accumulated_frame_count_, max_accumulated_frames_);
}


void PowerlineExtractor::processAndPublishPowerlines() {
    // 如果使用LiDAR数据但尚未接收到数据，则返回
    if (use_lidar_data_ && !new_lidar_data_available_) {
        ROS_INFO_THROTTLE(5.0, "Waiting for LiDAR data...");
        return;
    }
    
    // 如果使用.mat文件数据
    if (!use_lidar_data_) {
        // 转换数据为点云
        doubleToPointCloud();
    }
    
    // 发布原始点云
    sensor_msgs::PointCloud2 original_cloud_msg;
    pcl::toROSMsg(*original_cloud_, original_cloud_msg);
    original_cloud_msg.header.frame_id = target_frame_;
    original_cloud_msg.header.stamp = ros::Time::now();
    original_cloud_pub_.publish(original_cloud_msg);
    
    // 新增：裁减点云，只保留半径8m内的点
    clipPointCloud(original_cloud_, clipped_cloud_, clip_radius_);
    
    // 发布裁减后的点云
    sensor_msgs::PointCloud2 clipped_cloud_msg;
    pcl::toROSMsg(*clipped_cloud_, clipped_cloud_msg);
    clipped_cloud_msg.header.frame_id = target_frame_;
    clipped_cloud_msg.header.stamp = ros::Time::now();
    clipped_cloud_pub_.publish(clipped_cloud_msg);
    
    // 新增：累积裁减后的点云
    accumulatePointCloud(clipped_cloud_);
    
    // 发布累积点云
    sensor_msgs::PointCloud2 accumulated_cloud_msg;
    pcl::toROSMsg(*accumulated_cloud_, accumulated_cloud_msg);
    accumulated_cloud_msg.header.frame_id = target_frame_;
    accumulated_cloud_msg.header.stamp = ros::Time::now();
    accumulated_cloud_pub_.publish(accumulated_cloud_msg);
    
    // 对累积点云进行下采样，而不是原始点云
    downsamplePointCloud(accumulated_cloud_, downsampled_cloud);
    
    // 发布下采样后的点云
    sensor_msgs::PointCloud2 downsample_cloud_msg;
    pcl::toROSMsg(*downsampled_cloud, downsample_cloud_msg);
    downsample_cloud_msg.header.frame_id = target_frame_;
    downsample_cloud_msg.header.stamp = ros::Time::now();
    downsamole_cloud_pub_.publish(downsample_cloud_msg);
    
    
    extractNonGroundPoints();
    // 发布非地面点云
    sensor_msgs::PointCloud2 non_ground_msg;
    pcl::toROSMsg(*non_ground_cloud_, non_ground_msg);
    non_ground_msg.header.frame_id = target_frame_;
    non_ground_msg.header.stamp = ros::Time::now();
    non_ground_pub_.publish(non_ground_msg);
    
   // 提取电力线点
   extractPowerlinePoints();
    
   // 新增：对提取的电力线点进行后处理
   postProcessPowerlinePoints(powerline_cloud_);
    
    // 发布电力线点云
    sensor_msgs::PointCloud2 powerline_msg;
    pcl::toROSMsg(*powerline_cloud_, powerline_msg);
    powerline_msg.header.frame_id = target_frame_;
    powerline_msg.header.stamp = ros::Time::now();
    powerline_pub_.publish(powerline_msg);
    
    // 对电力线点进行聚类
    clusterPowerlines(powerline_cloud_, clustered_powerline_cloud_);
    
    // 发布聚类后的电力线点云
    sensor_msgs::PointCloud2 clustered_powerline_msg;
    pcl::toROSMsg(*clustered_powerline_cloud_, clustered_powerline_msg);
    clustered_powerline_msg.header.frame_id = target_frame_;
    clustered_powerline_msg.header.stamp = ros::Time::now();
    clustered_powerline_pub_.publish(clustered_powerline_msg);
    
    ROS_INFO("Processed and published point clouds: original(%zu), clipped(%zu), accumulated(%zu), "
             "downsampled(%zu), non-ground(%zu), powerline(%zu), clustered(%zu)",
             original_cloud_->size(), clipped_cloud_->size(), accumulated_cloud_->size(),
             downsampled_cloud->size(), non_ground_cloud_->size(), powerline_cloud_->size(), 
             clustered_powerline_cloud_->size());
             
    // 如果使用LiDAR数据，重置标志
    if (use_lidar_data_) {
        new_lidar_data_available_ = false;
    }
}