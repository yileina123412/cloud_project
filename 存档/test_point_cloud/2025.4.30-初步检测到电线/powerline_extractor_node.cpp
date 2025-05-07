#include "powerline_extractor.h"

int main(int argc, char** argv) {

    setlocale(LC_ALL, "");
    ros::init(argc, argv, "powerline_extractor_node");
    ros::NodeHandle nh;
    
    PowerlineExtractor extractor;
    
    // 设置发布频率
    ros::param::param<double>("~publish_rate", 1.0); // 默认1Hz
    double publish_rate;
    ros::param::get("~publish_rate", publish_rate);
    
    ros::Rate rate(publish_rate);
    
    while (ros::ok()) {
        // 处理回调和发布点云
        ros::spinOnce();
        extractor.processAndPublishPowerlines();
        rate.sleep();
    }
    
    return 0;
}