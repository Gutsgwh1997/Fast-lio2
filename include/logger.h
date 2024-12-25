#pragma once

#include <chrono>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef USE_ROS1
#include <ros/ros.h>
#define LOG_DEBUG(...) ROS_DEBUG(__VA_ARGS__)
#define LOG_INFO(...) ROS_INFO(__VA_ARGS__)
#define LOG_WARN(...) ROS_WARN(__VA_ARGS__)
#define LOG_ERROR(...) ROS_ERROR(__VA_ARGS__)
#else
#define LOG_DEBUG(...)
#define LOG_INFO(...)                                                                                    \
    std::cout << "[" << getTimeStamp() << "] " << getBasename(__FILE__) << ":" << __LINE__ << " [INFO] " \
              << formatString(__VA_ARGS__) << std::endl
#define LOG_WARN(...)                                                                                    \
    std::cout << "[" << getTimeStamp() << "] " << getBasename(__FILE__) << ":" << __LINE__ << " [WARN] " \
              << formatString(__VA_ARGS__) << std::endl
#define LOG_ERROR(...)                                                                                    \
    std::cout << "[" << getTimeStamp() << "] " << getBasename(__FILE__) << ":" << __LINE__ << " [ERROR] " \
              << formatString(__VA_ARGS__) << std::endl
#endif

inline std::string getTimeStamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return std::string(buf);
}

inline std::string getBasename(const char* filepath) {
    const char* base = std::strrchr(filepath, '/');
    return base ? std::string(base + 1) : std::string(filepath);
}

inline std::string formatString(const char* format, ...) {
    va_list args;
    va_start(args, format);

    auto doFormatting = [&](std::vector<char>& buf) {
        int size = std::vsnprintf(buf.data(), buf.size(), format, args);
        if (size < 0 || static_cast<size_t>(size) >= buf.size()) {
            return std::string("PRINTF_ERROR");
        }
        return std::string(buf.data(), size);
    };

    std::vector<char> buffer(128);
    while (true) {
        std::string result = doFormatting(buffer);
        if (!result.empty()) {
            va_end(args);
            return result;
        }
        buffer.resize(buffer.size() * 2);  // 空间不够就倍增扩大
    }
}