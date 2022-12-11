#pragma once

#include <opencv2/core.hpp>
#pragma comment(lib, "deltille.lib")

bool findDeltilleCorners(cv::Mat& input, cv::Size board_size, std::vector<cv::Point2f>& dest, const bool isDraw = false);
bool findSquareCorners(cv::Mat& input, cv::Size board_size, std::vector<cv::Point2f>& dest, const bool isDraw = false);
