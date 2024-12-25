#pragma once

#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace air_slam {
namespace FileSystem {
namespace fs = std::filesystem;
inline bool fileExists(const std::string &path) { return fs::exists(path); }

inline bool createDirectory(const std::string &path) {
    if (fs::exists(path)) {
        return true;
    }
    if (!fs::create_directories(path)) {
        return false;
    }
    return true;
}

inline bool CreateFile(std::ofstream &ofs, const std::string &file_path) {
    ofs.open(file_path, std::ios::out);
    if (!ofs) {
        return false;
    }
    return true;
}

inline std::string getTimeStamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d-%H-%M-%S", std::localtime(&t));
    return std::string(buf);
}

}  // namespace FileSystem

}  // namespace air_slam
