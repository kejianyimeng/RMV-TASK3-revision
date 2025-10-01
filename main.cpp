#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <ceres/ceres.h>

// 轨迹拟合的代价函数
struct TrajectoryCostFunction {
    TrajectoryCostFunction(double t, double x_obs, double y_obs, double x0, double y0)
        : t_(t), x_obs_(x_obs), y_obs_(y_obs), x0_(x0), y0_(y0) {}
    
    template<typename T>
    bool operator()(const T* const vx0, const T* const vy0, const T* const g, const T* const k, T* residual) const {
        T dt = T(t_);
        
        // x(t) = x0 + (vx0/k) * (1 - exp(-k*dt))
        T x_pred = T(x0_) + (vx0[0] / k[0]) * (T(1.0) - exp(-k[0] * dt));
        
        // y(t) = y0 + ((vy0 + g/k)/k) * (1 - exp(-k*dt)) - (g/k)*dt
        T term1 = (vy0[0] + g[0] / k[0]) / k[0];
        T term2 = T(1.0) - exp(-k[0] * dt);
        T term3 = (g[0] / k[0]) * dt;
        T y_pred = T(y0_) + term1 * term2 - term3;
        
        residual[0] = x_pred - T(x_obs_);
        residual[1] = y_pred - T(y_obs_);
        
        return true;
    }
    
private:
    double t_, x_obs_, y_obs_, x0_, y0_;
};

// 参数边界约束
class ParameterBoundary : public ceres::Manifold {
public:
    ParameterBoundary(double lower_bound, double upper_bound)
        : lower_bound_(lower_bound), upper_bound_(upper_bound) {}
    
    virtual int AmbientSize() const { return 1; }
    virtual int TangentSize() const { return 1; }
    
    virtual bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
        x_plus_delta[0] = x[0] + delta[0];
        return true;
    }
    
    virtual bool PlusJacobian(const double* x, double* jacobian) const {
        jacobian[0] = 1.0;
        return true;
    }
    
    virtual bool Minus(const double* y, const double* x, double* y_minus_x) const {
        y_minus_x[0] = y[0] - x[0];
        return true;
    }
    
    virtual bool MinusJacobian(const double* x, double* jacobian) const {
        jacobian[0] = 1.0;
        return true;
    }
    
    virtual bool ValidateParameters(const double* parameters) const {
        return (parameters[0] >= lower_bound_ && parameters[0] <= upper_bound_);
    }
    
private:
    double lower_bound_, upper_bound_;
};

// 从视频中提取蓝色小球轨迹点
std::vector<cv::Point2d> extractBlueBallPoints(const std::string& video_path) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频文件" << std::endl;
        return {};
    }
    
    std::vector<cv::Point2d> trajectory_points;
    cv::Mat frame, hsv_frame, mask;
    
    // 蓝色在HSV颜色空间中的范围
    cv::Scalar lower_blue(60, 60, 60);
    cv::Scalar upper_blue(130, 255, 255);
    
    // 获取视频的FPS信息
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 60.0;
    
    int frame_count = 0;
    double prev_x = -1, prev_y = -1;
    
    while (cap.read(frame)) {
        if (frame.empty()) break;
        
        // 转换为HSV颜色空间
        cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV);
        
        // 颜色阈值分割 - 检测蓝色
        cv::inRange(hsv_frame, lower_blue, upper_blue, mask);
        
        // 形态学操作去噪
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        
        // 查找轮廓
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        cv::Point2d current_point(-1, -1);
        bool point_found = false;
        
        if (!contours.empty()) {
            // 按面积排序，找到最大的几个轮廓
            std::sort(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                    return cv::contourArea(a) > cv::contourArea(b);
                });
            
            // 遍历前几个大轮廓，寻找圆形度高的
            for (size_t i = 0; i < std::min(contours.size(), size_t(3)); ++i) {
                double area = cv::contourArea(contours[i]);
                double perimeter = cv::arcLength(contours[i], true);
                
                // 计算圆形度
                double circularity = 0;
                if (perimeter > 0) {
                    circularity = (4 * CV_PI * area) / (perimeter * perimeter);
                }
                
                // 面积阈值和圆形度阈值
                if (area > 50 && area < 5000 && circularity > 0.7) {
                    cv::Moments m = cv::moments(contours[i]);
                    if (m.m00 > 0) {
                        double x = m.m10 / m.m00;
                        double y = m.m01 / m.m00;
                        
                        // 简单的运动连续性检查
                        if (prev_x >= 0 && prev_y >= 0) {
                            double distance = sqrt(pow(x - prev_x, 2) + pow(y - prev_y, 2));
                            if (distance > 100) {
                                continue;
                            }
                        }
                        
                        current_point = cv::Point2d(x, y);
                        point_found = true;
                        break;
                    }
                }
            }
        }
        
        if (point_found) {
            trajectory_points.push_back(current_point);
            prev_x = current_point.x;
            prev_y = current_point.y;
        }
        
        frame_count++;
    }
    
    cap.release();
    return trajectory_points;
}

// 轨迹拟合函数
void fitTrajectory(const std::vector<cv::Point2d>& data_points, double fps = 60.0) {
    if (data_points.size() < 4) {
        std::cerr << "提取到的数据点不足，无法进行轨迹拟合" << std::endl;
        return;
    }
    
    // 初始位置 (第一帧)
    double x0 = data_points[0].x;
    double y0 = data_points[0].y;
    
    // 初始参数估计
    double vx0 = 300.0;  // 初始x方向速度估计
    double vy0 = -50.0;  // 初始y方向速度估计
    double g = 500.0;    // 重力加速度初始估计
    double k = 0.1;      // 阻力系数初始估计
    
    // 创建优化问题
    ceres::Problem problem;
    
    // 添加参数边界约束
    problem.AddParameterBlock(&g, 1, new ParameterBoundary(100.0, 1000.0));
    problem.AddParameterBlock(&k, 1, new ParameterBoundary(0.01, 1.0));
    
    // 添加所有数据点的残差项
    for (size_t i = 1; i < data_points.size(); ++i) {
        double t = i / fps;  // 时间 (秒)
        double x_obs = data_points[i].x;
        double y_obs = data_points[i].y;
        
        ceres::CostFunction* cost_function = 
            new ceres::AutoDiffCostFunction<TrajectoryCostFunction, 2, 1, 1, 1, 1>(
                new TrajectoryCostFunction(t, x_obs, y_obs, x0, y0));
        
        problem.AddResidualBlock(cost_function, new ceres::HuberLoss(1.0), &vx0, &vy0, &g, &k);
    }
    
    // 配置求解器选项
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;  // 关闭中间输出
    options.max_num_iterations = 100;
    options.function_tolerance = 1e-8;
    options.parameter_tolerance = 1e-8;
    
    // 求解
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    
    // 计算拟合误差
    double total_error = 0.0;
    for (size_t i = 0; i < data_points.size(); ++i) {
        double t = i / fps;
        double dt = t;
        
        double x_pred = x0 + (vx0 / k) * (1.0 - exp(-k * dt));
        double term1 = (vy0 + g / k) / k;
        double term2 = 1.0 - exp(-k * dt);
        double term3 = (g / k) * dt;
        double y_pred = y0 + term1 * term2 - term3;
        
        double error_x = x_pred - data_points[i].x;
        double error_y = y_pred - data_points[i].y;
        total_error += sqrt(error_x * error_x + error_y * error_y);
    }
    
    double avg_error = total_error / data_points.size();
    
    // 输出最终结果
    std::cout << "=== 轨迹拟合结果 ===" << std::endl;
    std::cout << "提取数据点数量: " << data_points.size() << std::endl;
    std::cout << "初始速度 vx0 = " << vx0 << " px/s" << std::endl;
    std::cout << "初始速度 vy0 = " << vy0 << " px/s" << std::endl;
    std::cout << "重力加速度 g = " << g << " px/s²" << std::endl;
    std::cout << "阻力系数 k = " << k << " 1/s" << std::endl;
    std::cout << "初始位置 x0 = " << x0 << " px" << std::endl;
    std::cout << "初始位置 y0 = " << y0 << " px" << std::endl;
    std::cout << "平均拟合误差: " << avg_error << " px" << std::endl;
}

// 主函数
int main() {
    std::string video_path = "/home/xujiake/RM_Task/TASK3/video.mp4";
    
    // 从视频中提取蓝色小球轨迹点
    std::cout << "正在从视频中提取轨迹点..." << std::endl;
    auto points = extractBlueBallPoints(video_path);
    
    // 进行轨迹拟合
    fitTrajectory(points);
    
    return 0;
}