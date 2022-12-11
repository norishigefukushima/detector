//#include <deltille/DetectorTools.h>
#include <deltille/GridDetectorContext.h>
//#include <deltille/target_detector.h>

#include <opencp.hpp>//debug
using namespace cv;
using namespace std;

static void printBoard(string mes, Mat& board, vector<Point2f>& corner_locations)
{
	print_mat_format(board, mes, "%03d ");
	for (int j = 0; j < board.rows; j++)
	{
		for (int i = 0; i < board.cols; i++)
		{
			if (board.at<int>(j, i) == -1) printf("------------- ");
			else  printf("%6.1f-%6.1f ", corner_locations[board.cols * j + i].x, corner_locations[board.cols * j + i].y);
		}
		printf("\n");
	}
}

static void resetBoardIndex(Mat& destBoard)
{
	int idx = 0;
	for (int j = 0; j < destBoard.rows; j++)
	{
		for (int i = 0; i < destBoard.cols; i++)
		{
			if (destBoard.at<int>(j, i) != -1) destBoard.at<int>(j, i) = idx;
			idx++;
		}
	}
}

static void diagTransposeDeltille(Size board_size, Mat& board, vector<Point2f>& corner_locations)
{
	//printBoard("diag trans before", board, corner_locations);
	Mat boardC = board.clone();

	Mat destBoard(board_size.height + 2, board_size.width + 2 + (board_size.height - 1) / 2, CV_32S);
	destBoard.setTo(-1);
	vector<Point2f> destPoint(destBoard.size().area());
	for (int i = 0; i < destBoard.size().area(); i++)
	{
		destPoint[i] = Point2f(INFINITY, INFINITY);
	}

	int r = 0;
	vector<vector<Point2f>> points;
	for (int j = 1; j < board.rows - 1; j++)
	{
		for (int i = 1; i < board.cols - 1; i++)
		{
			vector<Point2f> v;
			if (boardC.at<int>(j, i) != -1)
			{
				for (int k = 0; k < board_size.width; k++)
				{
					const int bidx_x = i - k;
					const int bidx_y = j + k;
					const int didx_x = destBoard.cols - 2 - (r >> 1) - k;
					const int didx_y = r + 1;
					//print_debug4(bidx_y, bidx_x, didx_y, didx_x);
					destBoard.at<int>(didx_y, didx_x) = boardC.at<int>(bidx_y, bidx_x);
					v.push_back(corner_locations[bidx_y * boardC.cols + bidx_x]);
					boardC.at<int>(bidx_y, bidx_x) = -1;
				}
				r++;
				points.push_back(v);
			}
		}
	}

	bool hflag = points[0][0].x > points[0][board_size.width - 1].x;
	if (hflag)
	{
		//cout << "h flip" << endl;
		for (int j = 0; j < board_size.height; j++)
		{
			AutoBuffer<Point2f> tmp(board_size.width);
			for (int i = 0; i < board_size.width; i++)
			{
				tmp[board_size.width - 1 - i] = points[j][i];
			}
			for (int i = 0; i < board_size.width; i++)
			{
				points[j][i] = tmp[i];
			}
		}
	}

	bool isVflip = points[0][0].y > points[board_size.height - 1][0].y;
	int vidx = 1;
	for (int j = 0; j < board_size.height; j++)
	{
		for (int i = 0; i < board_size.width; i++)
		{
			const int dx = destBoard.cols - 1 - board_size.width - (vidx - 1) / 2 + i;
			Point2f pt = (isVflip) ? points[board_size.height - 1 - j][i] : points[j][i];
			destPoint[vidx * destBoard.cols + dx] = pt;
			destBoard.at<int>(vidx, dx) = 1;
		}
		vidx++;
	}

	destBoard.copyTo(board);
	resetBoardIndex(board);
	corner_locations.resize(0);
	for (int i = 0; i < destPoint.size(); i++)
	{
		corner_locations.push_back(destPoint[i]);
	}

	//printBoard("diag trans after", board, corner_locations);
}

static void transposeDeltille(Size board_size, Mat& board, vector<Point2f>& corner_locations)
{
	//printBoard("trans before", board, corner_locations);
	Mat destBoard(board_size.height + 2, board_size.width + 2 + (board_size.height - 1) / 2, CV_32S);
	destBoard.setTo(-1);
	vector<Point2f> destPoint(destBoard.total());
	for (int i = 0; i < destPoint.size(); i++)
	{
		destPoint[i] = Point2f(INFINITY, INFINITY);
	}

	vector<vector<Point2f>> points;
	for (int i = board.cols - 2; i >= 1; i--)
	{
		vector<Point2f> v;
		for (int j = 1; j < board.rows - 1; j++)
		{
			int val = board.at<int>(j, i);
			if (val != -1) v.push_back(corner_locations[j * board.cols + i]);
		}
		points.push_back(v);
	}

	bool hflag = points[0][0].x > points[0][board_size.width - 1].x;
	if (hflag)
	{
		//cout << "h flip" << endl;
		for (int j = 0; j < board_size.height; j++)
		{
			AutoBuffer<Point2f> tmp(board_size.width);
			for (int i = 0; i < board_size.width; i++)
			{
				tmp[board_size.width - 1 - i] = points[j][i];
			}
			for (int i = 0; i < board_size.width; i++)
			{
				points[j][i] = tmp[i];
			}
		}
	}

	bool isVflip = points[0][0].y > points[board_size.height - 1][0].y;

	int vidx = 1;
	for (int j = 0; j < board_size.height; j++)
	{
		for (int i = 0; i < board_size.width; i++)
		{
			const int dx = destBoard.cols - 1 - board_size.width - (vidx - 1) / 2 + i;
			if (isVflip)
			{
				destPoint[vidx * destBoard.cols + dx] = points[board_size.height - 1 - j][i];
				destBoard.at<int>(vidx, dx) = 1;
			}
			else
			{
				destPoint[vidx * destBoard.cols + dx] = points[j][i];
				destBoard.at<int>(vidx, dx) = 1;
			}

		}
		vidx++;
	}

	resetBoardIndex(destBoard);
	board = destBoard.clone();

	corner_locations.resize(0);
	for (int i = 0; i < destPoint.size(); i++)
	{
		corner_locations.push_back(destPoint[i]);
	}

	//printBoard("trans after", board, corner_locations);
}

static void flipDeltille(Mat& board, vector<Point2f>& corner_locations)
{
	//printBoard("flip before", board, corner_locations);
	bool isHFlip = false;
	bool isVFlip = false;
	Mat t = board.row(1);
	Mat b = board.row(board.rows - 2);
	//cout << t << endl;
	int maxv = 0;
	int argmax = 0;
	int minv = INT_MAX;
	int argmin = 0;

	int maxvb = 0;
	int argmaxb = 0;
	int minvb = INT_MAX;
	int argminb = 0;

	for (int i = 0; i < t.total(); i++)
	{
		if (t.at<int>(i) != -1)
		{
			if (maxv < t.at<int>(i))
			{
				maxv = t.at<int>(i);
				argmax = i;
			}
			if (minv > t.at<int>(i))
			{
				minv = t.at<int>(i);
				argmin = i;
			}
		}
	}

	for (int i = 0; i < b.total(); i++)
	{
		if (b.at<int>(i) != -1)
		{
			if (maxvb < b.at<int>(i))
			{
				maxvb = b.at<int>(i);
				argmaxb = i;
			}
			if (minvb > b.at<int>(i))
			{
				minvb = b.at<int>(i);
				argminb = i;
			}
		}
	}
	if (corner_locations[board.cols + argmin].y > corner_locations[board.cols * (board.rows - 2) + argminb].y)
	{
		isVFlip = true;
	}
	if (isVFlip) t = b;

	maxv = 0;
	argmax = 0;
	minv = INT_MAX;
	argmin = 0;
	for (int i = 0; i < t.total(); i++)
	{
		if (t.at<int>(i) != -1)
		{
			if (maxv < t.at<int>(i))
			{
				maxv = t.at<int>(i);
				argmax = i;
			}
			if (minv > t.at<int>(i))
			{
				minv = t.at<int>(i);
				argmin = i;
			}
		}
	}
	/*print_debug4(maxv, argmax, minv, argmin);
	print_debug(corner_locations[board.cols + argmin].x);
	print_debug(corner_locations[board.cols + argmax].x);
	*/
	if (isVFlip)
	{
		if (corner_locations[board.cols * (board.rows - 2) + argmin].x > corner_locations[board.cols * (board.rows - 2) + argmax].x)
		{
			isHFlip = true;
		}
	}
	else
	{
		if (corner_locations[board.cols + argmin].x > corner_locations[board.cols + argmax].x)
		{
			isHFlip = true;
		}
	}

	//flip
	Mat destBoard(board.size(), CV_32S); destBoard.setTo(-1);
	vector<Point2f> destPoint(corner_locations.size());
	if (isVFlip)
	{
		//cout << "v flip" << endl;
		for (int j = 1; j < board.rows - 1; j++)
		{
			Mat t = board.row(j);
			Mat b = board.row(board.rows - 1 - j);
			//cout << t << endl;
			int maxvt = 0;
			int argmaxt = 0;
			int minvt = INT_MAX;
			int argmint = 0;
			int maxvb = 0;
			int argmaxb = 0;
			int minvb = INT_MAX;
			int argminb = 0;

			for (int i = 0; i < t.total(); i++)
			{
				if (t.at<int>(i) != -1)
				{
					if (maxvt < t.at<int>(i))
					{
						maxvt = t.at<int>(i);
						argmaxt = i;
					}
					if (minvt > t.at<int>(i))
					{
						minvt = t.at<int>(i);
						argmint = i;
					}
				}
			}

			for (int i = 0; i < b.total(); i++)
			{
				if (b.at<int>(i) != -1)
				{
					if (maxvb < b.at<int>(i))
					{
						maxvb = b.at<int>(i);
						argmaxb = i;
					}
					if (minvb > b.at<int>(i))
					{
						minvb = b.at<int>(i);
						argminb = i;
					}
				}
			}
			//print_debug4(maxv, argmax, minv, argmin);
			if (argmint > argmaxt) swap(argmint, argmaxt);
			if (argminb > argmaxb) swap(argminb, argmaxb);

			for (int i = argmint; i <= argmaxt; i++)
			{
				destPoint[(board.rows - 1 - j) * board.cols + argminb + i - argmint] = corner_locations[j * board.cols + i];
				//	destBoard.at<int>(board.rows - 1 - j, argminb + i - argmint) = board.at<int>(j, i);
			}
		}

		for (int i = 0; i < corner_locations.size(); i++)
		{
			corner_locations[i] = destPoint[i];
		}
		//destBoard.copyTo(board);
	}
	if (isHFlip)
	{
		//print_mat_format(board, "beforeflip", "%03d ");
		for (int j = 1; j < board.rows - 1; j++)
		{
			Mat t = board.row(j);
			//cout << t << endl;
			int maxvt = 0;
			int argmaxt = 0;
			int minvt = INT_MAX;
			int argmint = 0;
			for (int i = 0; i < t.total(); i++)
			{
				if (t.at<int>(i) != -1)
				{
					if (maxvt < t.at<int>(i))
					{
						maxvt = t.at<int>(i);
						argmaxt = i;
					}
					if (minvt > t.at<int>(i))
					{
						minvt = t.at<int>(i);
						argmint = i;
					}
				}
			}
			//print_debug4(maxv, argmax, minv, argmin);
			if (argmint > argmaxt) swap(argmint, argmaxt);

			for (int i = argmint; i <= argmaxt; i++)
			{
				destPoint[j * board.cols + (argmaxt + argmint - i)] = corner_locations[j * board.cols + i];
				//	destBoard.at<int>(j, argmaxt + argmint - i) = board.at<int>(j, i);
			}
		}
		//cout << "h flip" << endl;
		corner_locations.resize(0);
		for (int i = 0; i < destPoint.size(); i++)
		{
			corner_locations.push_back(destPoint[i]);
		}
		//board = destBoard.clone();
	}

	bool isEvenOdd = true;
	for (int i = 0; i < board.cols; i++)
	{
		bool flag1 = board.at<int>(1, i) == -1;
		bool flag2 = board.at<int>(2, i) == -1;
		if (flag1 != flag2)isEvenOdd = false;
	}

	if (!isEvenOdd)
	{
		for (int j = 1; j < board.rows - 1; j++)
		{
			if (j % 2 == 0)
			{
				vector<Point2f> t(board.cols);
				for (int i = 0; i < board.cols; i++)
				{
					t[i] = corner_locations[j * board.cols + i];
				}
				for (int i = 0; i < board.cols; i++)
				{
					corner_locations[j * board.cols + i] = Point2f(INFINITY, INFINITY);
				}
				for (int i = 0; i < board.cols - 1; i++)
				{
					corner_locations[j * board.cols + i + 1] = t[i];
				}
				Mat a = board.row(j - 1);
				Mat b = board.row(j);
				a.copyTo(b);
			}
		}
	}
	resetBoardIndex(board);
	//printBoard("flip after", board, corner_locations);
}

static void numbering(Mat& input, Mat& board, vector<Point2f>& dest)
{
	int idx = 0;
	for (int i = 0; i < dest.size(); i++)
	{
		if (board.at<int>(i) != -1)
		{
			putText(input, format("%d", idx++), Point(dest[i]), FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, COLOR_RED);
		}
	}
}

static void reorderDeltilleCorners(Size board_size, Mat& board, vector<Point2f>& corner_locations)
{
	//if (board.size() == Size(board_size.width + board_size.height / 2 + 2, board_size.height + 2))// cout << "OK" << endl;

	vector<int> hcount;
	for (int i = 1; i < board.rows - 1; i++)
	{
		Mat sub = board.row(i);
		int count = 0;
		for (int i = 0; i < sub.total(); i++)
		{
			if (sub.at<int>(i) != -1)count++;
		}
		hcount.push_back(count);
	}
	vector<int> vcount;
	for (int i = 1; i < board.cols - 1; i++)
	{
		Mat sub = board.col(i);
		int count = 0;
		for (int i = 0; i < sub.total(); i++)
		{
			if (sub.at<int>(i) != -1)count++;
		}
		vcount.push_back(count);
	}

	int mode = 2;
	bool isTranspose = true;
	for (int i = 0; i < vcount.size() - 1; i++)
	{
		if (vcount[i] != vcount[i + 1])isTranspose = false;
	}
	if (isTranspose) mode = 1;

	bool isNone = true;
	for (int i = 0; i < hcount.size() - 1; i++)
	{
		if (hcount[i] != hcount[i + 1])isNone = false;
	}
	if (isNone) mode = 0;

	//print_debug(count);
	if (mode == 1)
	{
		//cout << "transpose" << endl;
		transposeDeltille(board_size, board, corner_locations);
	}
	else if (mode == 2)
	{
		//cout << "diag transpose" << endl;
		diagTransposeDeltille(board_size, board, corner_locations);
	}
	else
	{
		//cout << "no proc" << endl;
		flipDeltille(board, corner_locations);
		//printBoard("no proc", board, corner_locations);
	}
}

//#define DEBUG_DRAW

bool findDeltilleCorners(Mat& input, cv::Size board_size, vector<Point2f>& dest, const bool isDraw)
{
	std::vector<orp::calibration::BoardObservation> boards;
	//orp::calibration::GridDetectorContext<orp::calibration::MonkeySaddlePoint> grid_detector(input);
	//orp::calibration::GridDetectorContext<orp::calibration::SaddlePoint> grid_detector(input);
	orp::calibration::GridDetectorContext<orp::calibration::MonkeySaddlePoint, unsigned char, double> grid_detector(input);

	int ret = grid_detector.findBoards(board_size, boards, true);
#ifdef DEBUG_DRAW
	Mat before = input.clone();
	orp::calibration::drawCheckerboardCorners(before, boards[0], 2, true);
	numbering(before, boards[0].board, boards[0].corner_locations);
	imshow("before", before);
#endif
	if (boards.size() == 0) return false;;

	int count = 0;
	for (int i = 0; i < boards[0].board.total(); i++)
	{
		if (boards[0].board.at<int>(i) != -1)count++;
	}
	if (count != board_size.area())
	{
		return false;
	}

	reorderDeltilleCorners(board_size, boards[0].board, boards[0].corner_locations);

	//printBoard("trans after", boards[0].board, boards[0].corner_locations);
	//testImage = input.clone();
	if (isDraw && ret)
	{
		//print_mat_format(boards[0].board, "before", "%03d ");
		//reorderDeltilleCorners(board_size, boards[0].board, boards[0].corner_locations);
		//print_mat_format(boards[0].board, "after", "%03d ");
		orp::calibration::drawCheckerboardCorners(input, boards[0], 2, true);
		numbering(input, boards[0].board, boards[0].corner_locations);
	}

	dest.resize(0);
	for (int i = 0; i < boards[0].corner_locations.size(); i++)
	{
		if (boards[0].board.at<int>(i) != -1) dest.push_back(boards[0].corner_locations[i]);
	}
	if (dest.size() != board_size.area()) ret = false;
	return ret;
}


static void transposeSquare(cv::Size board_size, Mat& board, vector<Point2f>& points)
{
	vector<Point2f> tmp(points.size());
	for (int j = 0; j < board.rows; j++)
	{
		for (int i = 0; i < board.cols; i++)
		{
			tmp[board.rows * i + j] = points[board.cols * j + i];
		}
	}
	for (int i = 0; i < points.size(); i++)points[i] = tmp[i];
	transpose(board, board);
	resetBoardIndex(board);
}

static void flipSquare(cv::Size board_size, Mat& board, vector<Point2f>& points)
{
	vector<Point2f> tmp(points.size());
	const bool ishflip = (points[board.cols * 1 + 1].x > points[board.cols * 1 + board.cols - 2].x);
	if (ishflip)
	{
		for (int j = 0; j < board.rows; j++)
		{
			for (int i = 0; i < board.cols; i++)
			{
				tmp[board.cols * j + (board.cols - 1 - i)] = points[board.cols * j + i];
			}
		}
		for (int i = 0; i < points.size(); i++)points[i] = tmp[i];
	}

	const bool isvflip = (points[board.cols * 1 + 1].y > points[board.cols * (board.rows - 2) + 1].y);
	if (isvflip)
	{
		for (int j = 0; j < board.rows; j++)
		{
			for (int i = 0; i < board.cols; i++)
			{
				tmp[board.cols * (board.rows - 1 - j) + i] = points[board.cols * j + i];
			}
		}
		for (int i = 0; i < points.size(); i++)points[i] = tmp[i];
	}
}

bool findSquareCorners(Mat& input, cv::Size board_size, vector<Point2f>& dest, const bool isDraw)
{
	std::vector<orp::calibration::BoardObservation> boards;
	//orp::calibration::GridDetectorContext<orp::calibration::MonkeySaddlePoint> grid_detector(input);
	orp::calibration::GridDetectorContext<orp::calibration::SaddlePoint, unsigned char, double> grid_detector(input);
	//orp::calibration::GridDetectorContext<orp::calibration::MonkeySaddlePoint, unsigned char, double> grid_detector(input);

	int ret = grid_detector.findBoards(board_size, boards, true);
#ifdef DEBUG_DRAW
	Mat before = input.clone();
	orp::calibration::drawCheckerboardCorners(before, boards[0], 2, true);
	numbering(before, boards[0].board, boards[0].corner_locations);
	imshow("before", before);
#endif
	if (boards.size() == 0) return false;;

	int count = 0;
	for (int i = 0; i < boards[0].board.total(); i++)
	{
		if (boards[0].board.at<int>(i) != -1)count++;
	}
	if (count != board_size.area())
	{
		return false;
	}

	if (boards[0].board.cols - 2 != board_size.width)
	{
		transposeSquare(board_size, boards[0].board, boards[0].corner_locations);
	}
	flipSquare(board_size, boards[0].board, boards[0].corner_locations);
	//reorderDeltilleCorners(board_size, boards[0].board, boards[0].corner_locations);

	//printBoard("trans after", boards[0].board, boards[0].corner_locations);
	//testImage = input.clone();
	if (isDraw && ret)
	{
		//print_mat_format(boards[0].board, "before", "%03d ");
		//reorderDeltilleCorners(board_size, boards[0].board, boards[0].corner_locations);
		//print_mat_format(boards[0].board, "after", "%03d ");
		orp::calibration::drawCheckerboardCorners(input, boards[0], 2, true);
		numbering(input, boards[0].board, boards[0].corner_locations);
	}

	dest.resize(0);
	for (int i = 0; i < boards[0].corner_locations.size(); i++)
	{
		if (boards[0].board.at<int>(i) != -1) dest.push_back(boards[0].corner_locations[i]);
	}
	if (dest.size() != board_size.area()) ret = false;
	return ret;
}