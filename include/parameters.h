// Author: qingShan

#pragma once

#include <vector>

namespace air_slam
{

    struct LioParams
    {
        double resolution = 0.1;
        double esikf_min_iteration = 2;
        double esikf_max_iteration = 5;
        double imu_acc_cov = 0.01;
        double imu_gyro_cov = 0.01;
        double imu_acc_bias_cov = 0.0001;
        double imu_gyro_bias_cov = 0.0001;

        std::vector<double> imu_ext_rot = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        std::vector<double> imu_ext_pos = {-0.011, -0.02329, 0.04412};

        double cube_len = 1000.0;
        double det_range = 100.0;
        double move_thresh = 1.5;

        bool extrinsic_est_en = false;
        bool align_gravity = false;
    };

} // namespace air_slam