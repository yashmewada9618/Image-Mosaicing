#include <cstdio>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/*
    Author - Yash Mewada (mewada.y@northeastern.edu) & Pratik Baldota (baldota.p@northeastern.edu)
    Created - Feb 23, 2023
*/

/* Class for Image Mosaicing function declaration*/
class Image_Mosaicing
{
private:
    /* data */
public:
    string path_to_images;
    Mat img1;
    Mat img2;
    Mat img1_gray;
    Mat img2_gray;
    Mat soble_x = (Mat_<float>(3,3) << -1,0,1,-2,0,2,-1,0,1);
    Mat soble_y = (Mat_<float>(3,3) << -1,-2,-1,0,0,0,1,2,1);
    const int MIN_POINTS = 4;
    const double THRESHOLD = 10;
    const int MAX_ITERATIONS = 1000;
    double ncc_thres = 0.9;
    Image_Mosaicing(string _path);
    
    pair<vector<Point>, vector<Point>> perform_harris(int thresh);
    
    double calc_NCC(Mat temp1, Mat temp2);
    
    vector<pair<Point, Point>> get_correspondences(vector<Point> c1,vector<Point> c2);
    
    void visualise_corress(vector<pair<Point, Point>> corresspondences);
    
    void compute_homography(Mat matched_corners1, Mat mathched_corners2);
    
    vector<Point> get_random_points(vector<Point> points, int n);
    
    Mat compute_homography(vector<Point> src_points, vector<Point> dst_points);
    
    vector<int> get_inliers(vector<Point> src_points, vector<Point> dst_points, Mat homography);

    Mat estimate_homography_ransac(vector<Point> src_points, vector<Point> dst_points);
    vector<Point> harris_detector_for_img1(int thres = 250);
    vector<Point> harris_detector_for_img2(int thres = 250);
    vector<Point2f> cvt_pts_pt2f(vector<Point> points);
    Mat find_ovelapping_region(Mat best_homography);
    Mat alpha_blending(Mat image1, Mat image2,double alpha);
    // vector<Point> getMouseClicks(int event, int x, int y, int flags, void* userdata);

};
Image_Mosaicing::Image_Mosaicing(string _path)
{
    cout << "This is a demo for Image Mosaicing code" << endl;
    this->path_to_images = _path; 
    img1 = imread(_path + string("DSC_0309.JPG"));
    img2 = imread(_path + string("DSC_0310.JPG"));
    // resize(img1, img1, Size(), 0.75, 0.75);
    // resize(img2, img2, Size(), 0.75, 0.75);
    cvtColor(img1, img1_gray, COLOR_BGR2GRAY);
    cvtColor(img2, img2_gray, COLOR_BGR2GRAY);
}


/* Apply sobel mask to the images and Compute the harris R function along with the detected corners*/

vector<Point> Image_Mosaicing::harris_detector_for_img1(int thres){
    Mat gradient_x, gx, gxy;
    Mat gradient_y, gy;
    Mat r_norm;
    filter2D(img1_gray,gradient_x,CV_32F,soble_x);
    filter2D(img1_gray,gradient_y,CV_32F,soble_y);
    gx = gradient_x.mul(gradient_x);
    gy = gradient_y.mul(gradient_y);
    gxy = gradient_x.mul(gradient_y);
    GaussianBlur(gx,gx,Size(5,5),1.4);
    GaussianBlur(gy,gy,Size(5,5),1.4);
    GaussianBlur(gxy,gxy,Size(5,5),1.4);
    Mat r = Mat::zeros(img1_gray.size(), CV_32FC1);
    for(int i = 0; i < img1_gray.rows; i++){
        for(int j = 0; j < img1_gray.cols; j++){
            float a = gx.at<float>(i, j);
            float b = gy.at<float>(i, j);
            float c = gxy.at<float>(i, j);
            float det = a*c - b*b;
            float trace = a + c;
            r.at<float>(i,j) = det - 0.04*trace*trace;
        }
    }
    normalize(r, r_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat()); 
    // convertScaleAbs( r_norm, r_scaled );
    // Mat corner_img = img1.clone();
    // cvtColor(corner_img,corner_img,COLOR_GRAY2BGR);
    Mat corners = Mat::zeros(img1_gray.size(),CV_8UC1);
    vector<Point> corner_coor;
    // cvtColor(img1,img1,COLOR_GRAY2BGR);
     for (int i = 1; i < r_norm.rows; i++) {
        for (int j = 1; j < r_norm.cols; j++) {
            // Check if current pixel is a local maximum
            // cout << r_norm.at<float>(i, j) << "int";
            cout << (int) r_norm.at<float>(i, j) << endl;
            if ((int) r_norm.at<float>(i, j) > thres
                && r_norm.at<float>(i, j) > r_norm.at<float>(i - 1, j - 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i - 1, j)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i - 1, j + 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i, j - 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i, j + 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i + 1, j - 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i + 1, j)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i + 1, j + 1)
                ) {
                // Set pixel as corner
                // corners.at<uchar>(i, j) = 255;
                corner_coor.push_back({j,i});
                circle(img1, Point(j,i), 1, Scalar(0, 0, 255), 2,8,0);
            }
        }
    }
    cout << corner_coor.size() << endl;
    cv::imshow("corners",img1);
    // cv::imwrite("corner_2.jpg", img2);
    cv::waitKey(0);
    return corner_coor;
}
vector<Point> Image_Mosaicing::harris_detector_for_img2(int thres){
    Mat gradient_x, gx2, gxy;
    Mat gradient_y, gy2;
    Mat r_norm;
    filter2D(img2_gray,gradient_x,CV_32F,soble_x);
    filter2D(img2_gray,gradient_y,CV_32F,soble_y);
    gx2 = gradient_x.mul(gradient_x);
    gy2 = gradient_y.mul(gradient_y);
    gxy = gradient_x.mul(gradient_y);
    GaussianBlur(gx2,gx2,Size(5,5),1.4);
    GaussianBlur(gy2,gy2,Size(5,5),1.4);
    GaussianBlur(gxy,gxy,Size(5,5),1.4);
    Mat r = Mat::zeros(img2_gray.size(), CV_32FC1);
    for(int i = 0; i < img2_gray.rows; i++){
        for(int j = 0; j < img2_gray.cols; j++){
            float a = gx2.at<float>(i, j);
            float b = gy2.at<float>(i, j);
            float c = gxy.at<float>(i, j);
            float det = a*c - b*b;
            float trace = a + c;
            r.at<float>(i,j) = det - 0.04*trace*trace;
        }
    }
    normalize(r, r_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat()); 
    // convertScaleAbs( r_norm, r_scaled );
    // Mat corner_img = img1.clone();
    // cvtColor(corner_img,corner_img,COLOR_GRAY2BGR);
    Mat corners = Mat::zeros(img2_gray.size(),CV_8UC1);
    vector<Point> corner_coor;
    // cvtColor(img1,img1,COLOR_GRAY2BGR);
     for (int i = 1; i < r_norm.rows; i++) {
        for (int j = 1; j < r_norm.cols; j++) {
            // Check if current pixel is a local maximum
            // cout << r_norm.at<float>(i, j) << "int";
            // cout << (int) r_norm.at<float>(i, j) << endl;
            if ((int) r_norm.at<float>(i, j) > thres
                && r_norm.at<float>(i, j) > r_norm.at<float>(i - 1, j - 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i - 1, j)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i - 1, j + 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i, j - 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i, j + 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i + 1, j - 1)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i + 1, j)
                && r_norm.at<float>(i, j) > r_norm.at<float>(i + 1, j + 1)
                ) {
                // Set pixel as corner
                // corners.at<uchar>(i, j) = 255;
                corner_coor.push_back({j,i});
                circle(img2, Point(j,i), 1, Scalar(0, 0, 255), 2,8,0);
            }
        }
    }
    cout << corner_coor.size() << endl;
    cv::imshow("corners",img2);
    // cv::imwrite("corner_2.jpg", img2);
    cv::waitKey(0);
    return corner_coor;
}
pair<vector<Point>, vector<Point>> Image_Mosaicing::perform_harris(int thresh){
    Mat dst, dst_norm, dst_norm_scaled;
    Mat dst2, dst_norm2, dst_norm_scaled2;
    vector<Point> cor_1,cor_2;
    dst = Mat::zeros(img1_gray.size(), CV_32FC1);
    dst2 = Mat::zeros(img2_gray.size(), CV_32FC1);

    int blockSize = 2;
    int apertureSize = 5;
    double k = 0.04;

    cornerHarris(img1_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat()); 
    convertScaleAbs( dst_norm, dst_norm_scaled );

    cornerHarris(img2_gray, dst2, blockSize, apertureSize, k, BORDER_DEFAULT);
    normalize(dst2, dst_norm2, 0, 255, NORM_MINMAX, CV_32FC1, Mat()); 
    convertScaleAbs( dst_norm2, dst_norm_scaled2 );
    
    vector<Point> corner_coor;

    for( int i = 0; i < dst_norm.rows ; i++ )
    {
        for( int j = 0; j < dst_norm.cols; j++ )
        {
            if( (int) dst_norm.at<float>(i,j) > thresh
            && dst_norm.at<float>(i, j) > dst_norm.at<float>(i - 1, j - 1)
            && dst_norm.at<float>(i, j) > dst_norm.at<float>(i - 1, j)
            && dst_norm.at<float>(i, j) > dst_norm.at<float>(i - 1, j + 1)
            && dst_norm.at<float>(i, j) > dst_norm.at<float>(i, j - 1)
            && dst_norm.at<float>(i, j) > dst_norm.at<float>(i, j + 1)
            && dst_norm.at<float>(i, j) > dst_norm.at<float>(i + 1, j - 1)
            && dst_norm.at<float>(i, j) > dst_norm.at<float>(i + 1, j)
            && dst_norm.at<float>(i, j) > dst_norm.at<float>(i + 1, j + 1)
            )
            {
                circle( img1, Point(j,i), 1, Scalar(255,0,0), 2, 8, 0 );
                cor_1.push_back(Point(j,i));
            }
        }
    }
    for( int i = 0; i < dst_norm2.rows ; i++ )
    {
        for( int j = 0; j < dst_norm2.cols; j++ )
        {
            if( (int) dst_norm2.at<float>(i,j) > thresh
            && dst_norm2.at<float>(i, j) > dst_norm2.at<float>(i - 1, j - 1)
            && dst_norm2.at<float>(i, j) > dst_norm2.at<float>(i - 1, j)
            && dst_norm2.at<float>(i, j) > dst_norm2.at<float>(i - 1, j + 1)
            && dst_norm2.at<float>(i, j) > dst_norm2.at<float>(i, j - 1)
            && dst_norm2.at<float>(i, j) > dst_norm2.at<float>(i, j + 1)
            && dst_norm2.at<float>(i, j) > dst_norm2.at<float>(i + 1, j - 1)
            && dst_norm2.at<float>(i, j) > dst_norm2.at<float>(i + 1, j)
            && dst_norm2.at<float>(i, j) > dst_norm2.at<float>(i + 1, j + 1)
            )
            {
                circle( img2, Point(j,i), 1, Scalar(255,0,0), 2, 8, 0 );
                cor_2.push_back(Point(j,i));
                // cout << Point(j,i) << endl;
            }
        }
    }
    Mat concated_img;
    cout << "corner1_size" << "  ";
    cout << cor_1.size() << "   ";
    cout << "corner2_size" << "  ";
    cout << cor_2.size() << endl;
    if (img1.cols != img2.cols) {
        double scale = (double)img1.cols / img2.cols;
        resize(img2, img2, Size(img1.cols, scale*img2.rows));
    }
    
    // Concatenate images vertically
    Mat result;
    cv::vconcat(img1, img2, concated_img);
    cv::namedWindow( "corners_window" );
    cv::imshow( "corners_window", concated_img);
    cv::imwrite( "corners_window.jpg", concated_img);
    cv::waitKey(0);
    return make_pair(cor_1,cor_2);
}

double Image_Mosaicing::calc_NCC(Mat temp1,Mat temp2){
    double mean1 = 0;
    for(int i=0; i<temp1.rows; i++)
    {
        for(int j=0; j<temp1.cols; j++)
        {
            mean1 += temp1.at<uchar>(i,j);
        }
    }
    mean1 = mean1/(temp1.rows*temp1.cols);
    double mean2 = 0;
    for(int i=0; i<temp2.rows; i++)
    {
        for(int j=0; j<temp2.cols; j++)
        {
            mean2 += temp2.at<uchar>(i,j);
        }
    }
    mean2 = mean2/(temp2.rows*temp2.cols);
    double std1 = 0;
    for(int i=0; i<temp1.rows; i++)
    {
        for(int j=0; j<temp1.cols; j++)
        {
            std1 += pow(temp1.at<uchar>(i,j) - mean1, 2);
        }
    }
    std1 = sqrt(std1/(temp1.rows*temp1.cols));
    double std2 = 0;
    for(int i=0; i<temp2.rows; i++)
    {
        for(int j=0; j<temp2.cols; j++)
        {
            std2 += pow(temp2.at<uchar>(i,j) - mean2, 2);
        }
    }
    std2 = sqrt(std2/(temp2.rows*temp2.cols));
    double ncc = 0;
    // int count = 0;
    for(int i=0; i<temp1.rows; i++)
    {
        for(int j=0; j<temp1.cols; j++)
        {
            ncc += (temp1.at<uchar>(i,j) - mean1)*(temp2.at<uchar>(i,j) - mean2);
            // count++;
        }
    }
    if (std1 > 0 && std2 > 0) {
        ncc = ncc/(temp1.rows*temp1.cols*std1*std2);
    } 
    else {
        ncc = 0; // or set to some other default value
    }
    // ncc = ncc/(temp1.rows*temp1.cols*std1*std2);
    return ncc;

}

vector<pair<Point, Point>> Image_Mosaicing::get_correspondences(vector<Point> c1,vector<Point> c2){
    Mat t1,t2;
    vector<pair<Point,Point>> corres;
    Mat temp_path,temp_path2;
    Point d = Point(0,0);

    for (int i = 0; i < c1.size() ; i++) {
        double ncc_max = 0;
        Point pt1 = c1[i];
        int p1x = pt1.x - 3;
        int p1y = pt1.y - 3;
        if (p1x < 0 || p1y < 0 || p1x + 7 >= img1.cols || p1y + 7 >= img1.rows){
            continue;
        }
        temp_path = img1(Rect(p1x, p1y, 7, 7));
        d = Point(0,0);
        int maxidx = -1;
        for (int j = 0; j < c2.size(); j++) {
            Point pt2 = c2[j];
            int p2x = pt2.x - 3;
            int p2y = pt2.y - 3;
            if (p2x < 0 || p2y < 0 || p2x + 7 >= img2.cols || p2y + 7 >= img2.rows){
                continue;
            }
            temp_path2 = img2(Rect(p2x,p2y, 7, 7));

            double temp_ncc = calc_NCC(temp_path,temp_path2);
            if (temp_ncc > ncc_max){
                ncc_max = temp_ncc;
                maxidx = j;
            }
        }
        if (c2[maxidx] != Point(0,0) && c1[i] != Point(0,0) && ncc_max > ncc_thres){
            pair<Point,Point> c;
            c.first = c1[i];
            c.second = c2[maxidx]; 
            cout << "maxidx" << " ";
            cout << maxidx << endl;
            corres.push_back(c);           
        }
    }
    cout << corres.size() << endl;
    Mat img_matches;
    if (img1.cols != img2.cols) {
        double scale = (double)img1.cols / img2.cols;
        resize(img2, img2, Size(img1.cols, scale*img2.rows));
    }
    
    // Concatenate images vertically
    // vconcat(img1, img2, img_matches);
    hconcat(img1, img2, img_matches);
    for (int i = 0; i < corres.size() ; i++) {
        Point pt1 =  corres[i].first;
        Point pt2 = Point(corres[i].second.x + img1.cols, corres[i].second.y); // shift the x-coordinate of pt2 to the right of img1
        // Point pt2 = Point(fc[i].second.x, fc[i].second.y + img1.rows);
        line(img_matches, pt1, pt2, Scalar(0, 255, 0), 1);
    }
    imshow( "result_window", img_matches );
    cv::imwrite("CorrepondencespreHomography.jpg",img_matches);
    cv::waitKey(0);
    return corres;
}

void Image_Mosaicing::visualise_corress(vector<pair<Point, Point>> fc){
    Mat img_matches;
    if (img1.cols != img2.cols) {
        double scale = (double)img1.cols / img2.cols;
        resize(img2, img2, Size(img1.cols, scale*img2.rows));
    }
    
    // Concatenate images vertically
    // vconcat(img1, img2, img_matches);
    hconcat(img1, img2, img_matches);
    for (int i = 0; i < fc.size() ; i++) {
        Point pt1 =  fc[i].first;
        Point pt2 = Point(fc[i].second.x + img1.cols, fc[i].second.y); // shift the x-coordinate of pt2 to the right of img1
        // Point pt2 = Point(fc[i].second.x, fc[i].second.y + img1.rows);
        line(img_matches, pt1, pt2, Scalar(0, 255, 0), 1);
    }
    imshow( "result_window", img_matches );
    cv::imwrite("CorrepondencespreHomography.jpg",img_matches);
    cv::waitKey(0);
}

vector<Point> Image_Mosaicing::get_random_points(vector<Point> points, int n){
    // random_shuffle(points.begin(), points.end());
    vector<Point> random_points;
    // cout << points << endl;
    for (int i = 0; i < n; i++) {
        int random_num = rand() % points.size();
        random_points.push_back(points[random_num]);
    }
    return random_points;
}

Mat Image_Mosaicing::compute_homography(vector<Point> src_points, vector<Point> dst_points) {
    Mat homography = findHomography(src_points, dst_points, RANSAC, THRESHOLD);
    // cout << "found homography" << endl;
    // create a matrix of 8x9
    return homography;
}

vector<Point2f> Image_Mosaicing::cvt_pts_pt2f(vector<Point> points){
    vector<Point2f> new_pts;
    for (int i = 0; i < points.size(); i++){
        new_pts.push_back(Point2f(points[i].x,points[i].y));
    }
    return new_pts;
}

Mat Image_Mosaicing::estimate_homography_ransac(vector<Point> src_points, vector<Point> dst_points) {
    vector<pair<Point, Point>> bestCorrespondingPoints;
    int max_inliers = 0;
    // src_points.resize(src_points.size());
    vector<int> best_inliers;
    vector<Point> inliers1;
    vector<Point> inliers2;
    Mat best_homography = Mat::eye(3, 3, CV_64F);
    best_homography =findHomography(cvt_pts_pt2f(src_points),cvt_pts_pt2f(dst_points),RANSAC);
    // int n = 0;
    // cout << src_points.size() << endl;
    // cout << dst_points.size() << endl;

    // srand(time(NULL));
    // for (int i = 0; i < MAX_ITERATIONS; i++) {
    //     // cout << i << endl;
    //     vector<Point> random_src_points = get_random_points(src_points, MIN_POINTS);
    //     vector<Point> random_dst_points = get_random_points(dst_points, MIN_POINTS);
    //     Mat homography = findHomography(cvt_pts_pt2f(random_src_points),cvt_pts_pt2f(random_dst_points),RANSAC);
    //     vector<Point> inliers1_tmp, inliers2_tmp;
    //     // cout << homography << endl;
    //     if (homography.empty()) {
    //         continue;
    //     }
    //     // vector<int> inliers = get_inliers(src_points, dst_points, homography);
        vector<int> inliers;
        int num_inliers = 0;
        vector<pair<Point, Point>> temp_corres;
        int inlier_idx = -1;
    //     vector<Point2f> curr_inliers;
        for (int j = 0; j < dst_points.size(); j++) {
            Point src_point = src_points[j];
            Point dst_point = dst_points[j];

            Mat src = (Mat_<double>(3, 1) << src_point.x, src_point.y, 1);
            Mat dst = (Mat_<double>(3, 1) << dst_point.x, dst_point.y, 1);

            Mat pred_dst = best_homography * src;
            pred_dst /= pred_dst.at<double>(2, 0);
            
            double distance = norm(pred_dst-dst);
            // cout << src <<" << norm , manual >> ";SSSSS
            // cout << p << endl;
            cout << distance << endl;
            if (distance < 1) {
                // cout << "got" << endl;
                num_inliers++;
                // inlier_idx = j;
                pair<Point,Point> c;
                c.first = src_point;
                c.second = dst_point; 
                temp_corres.push_back(c);
                inliers.push_back(j);
                // cout << num_inliers << endl;
                // curr_inliers.push_back(src_points[j]);
            }
        if (num_inliers >= max_inliers) {
            cout << "blahh" << endl;
            max_inliers = num_inliers;
            best_inliers = inliers;
            // best_inliers = curr_inliers;
            best_homography = best_homography;
            bestCorrespondingPoints = temp_corres;
            // best_homography = estimateAffinePartial2D(src_points, dst_points, inliers, RANSAC, THRESHOLD);
        }
    }
    // cout << "max_inliers" << "    ";
    // cout << max_inliers << endl;
    vector<Point> inlier_src_points;
    vector<Point> inlier_dst_points;
    for (int i = 0; i < best_inliers.size(); i++) {
        int idx = best_inliers[i];
        inlier_src_points.push_back(src_points[idx]);
        inlier_dst_points.push_back(dst_points[idx]);
        cout << idx << endl;
    }
    // cout << best_homography << endl;
    best_homography = findHomography(cvt_pts_pt2f(inlier_src_points),cvt_pts_pt2f(inlier_dst_points),RANSAC);
    Mat img_matches;
    if (img1.cols != img2.cols) {
        double scale = (double)img1.cols / img2.cols;
        resize(img2, img2, Size(img1.cols, scale*img2.rows));
    }
    
    // Concatenate images vertically
    // vconcat(img1, img2, img_matches);
    hconcat(img1, img2, img_matches);
    for (int i = 0; i < bestCorrespondingPoints.size() ; i++) {
        Point pt1 =  bestCorrespondingPoints[i].first;
        Point pt2 = Point(bestCorrespondingPoints[i].second.x + img1.cols, bestCorrespondingPoints[i].second.y); // shift the x-coordinate of pt2 to the right of img1
        // Point pt2 = Point(fc[i].second.x, fc[i].second.y + img1.rows);
        line(img_matches, pt1, pt2, Scalar(0, 255, 0), 1);
    }
    imshow( "result_window", img_matches );
    cv::imwrite("CorrepondencespostHomography.jpg",img_matches);
    cv::waitKey(0);
    // visualise_corress(bestCorrespondingPoints);
    
    return best_homography;
}

Mat Image_Mosaicing::find_ovelapping_region(Mat best_homography){
    vector<Point> c1,c2;
    c1.push_back(Point(0,0));
    c1.push_back(Point(0,img1.rows));
    c1.push_back(Point(img1.cols,0));
    c1.push_back(Point(img1.cols,img1.rows));

    for (auto e : c1){
        Mat src = (Mat_<double>(3, 1) << e.x, e.y, 1);
        Mat pred_dst = best_homography * src;
        cout << pred_dst << "pred_dts";
        cout << e << endl;
        pred_dst /= pred_dst.at<double>(2, 0);
        c2.push_back(Point(pred_dst.at<double>(0,0),pred_dst.at<double>(1,0)));
    }
    int minx = min(min(c2[0].x,c2[1].x),min(c2[2].x,c2[3].x));
    int maxx = max(max(c2[0].x,c2[1].x),max(c2[2].x,c2[3].x));
    int miny = min(min(c2[0].y,c2[1].y),min(c2[2].y,c2[3].y));
    int maxy = max(max(c2[0].y,c2[1].y),max(c2[2].y,c2[3].y));
    cout << minx << endl;
    cout << maxx << endl;
    cout << abs(miny) << endl;
    cout << maxy << endl;
}

Mat Image_Mosaicing::alpha_blending(Mat image1, Mat image2, double alpha){
    // Mat alpha_img = image2.clone();
    // image1.convertTo(image1,CV_32FC3);
    // image2.convertTo(image2,CV_32FC3);

    // alpha_img.convertTo(alpha_img,CV_32FC3,1.0/255);

    // Mat out_img = Mat::zeros(image1.size(),image1.type());
    // multiply(alpha_img,image1,image1);
    // multiply(Scalar::all(1.0)-alpha_img,image2,image2);

    // add(image1,image2,out_img);

    // return out_img;
    // double beta;
    // Mat dst;
    // Mat chopped1 = image1(Rect(0, 0, 371, 340));
    // Mat chopped2 = image1(Rect(141, 0, 371, 340));
    // Mat avg = image1.clone();
    int min_y = 26;
    int min_x = 141;
    int max_x = 701;
    int max_y = 380;
    // Mat distances(image1.rows, image1.cols, CV_32F);
    // for (int i = 0; i < image1.rows; i++)
    // {
    //     for (int j = 0; j < image1.cols; j++)
    //     {
    //         float dist = std::min({i - min_y, max_y - i, j - min_x, max_x - j});
    //         distances.at<float>(i, j) = dist;
    //     }
    // }

    // Mat weights(image1.rows, image1.cols, CV_32F);
    // float max_dist = std::max(max_y - min_y, max_x - min_x);
    // for (int i = 0; i < image1.rows; i++)
    // {
    //     for (int j = 0; j < image1.cols; j++)
    //     {
    //         float dist = distances.at<float>(i, j);
    //         weights.at<float>(i, j) = 1.0 - dist / max_dist;
    //     }
    // }
    // // vector<Vec3b> p3;
    Vec3b sum(0, 0, 0);
    int count = 0;
    for (int i = min_y; i < max_y; i++)
    {   
        for (int j = min_x; j < max_x; j++)
        {

            if(i >= 0 && i < image1.rows && j >=0 && j < image1.cols){
                Vec3b avg_pix = image1.at<Vec3b>(i,j);
                sum += avg_pix;
                count++;
                cout << sum << endl;
            }
            // Vec3b p1 = image1.at<Vec3b>(i,j);
            // Vec3b p2 = image1.at<Vec3b>(i,j);
            // Vec3b p4 = (p1 + p2)/2;
            // image1.at<Vec3b>(i,j) = p4/2;
            // p3.push_back(p4);
        }
    }
    
    Vec3b avgerage = sum / count;
    int offset = 0;
    avgerage[0] = std::min(255, avgerage[0] + offset);  // blue channel
    avgerage[1] = std::min(255, avgerage[1] + offset);  // green channel
    avgerage[2] = std::min(255, avgerage[2] + offset);  // red channel
    for (int i = min_y; i < max_y; i++)
    {   
        for (int j = min_x; j < max_x; j++)
        {


            if(i >= 0 && i < image1.rows && j >=0 && j < image1.cols){
                image1.at<Vec3b>(i,j) = avgerage;
            }
        }
    }
    // cout << " Simple Linear Blender " << endl;
    // cout << "-----------------------" << endl;
    // cout << "* Enter alpha [0.0-1.0]: ";
    // cin >> input;
    // // We use the alpha provided by the user if it is between 0 and 1
    // if( input >= 0 && input <= 1 )
    //   { alpha = input; }
    // src1 = imread( samples::findFile("LinuxLogo.jpg") );
    // src2 = imread( samples::findFile("WindowsLogo.jpg") );
    // if( src1.empty() ) { cout << "Error loading src1" << endl; return EXIT_FAILURE; }
    // if( src2.empty() ) { cout << "Error loading src2" << endl; return EXIT_FAILURE; }
    // beta = ( 1.0 - alpha );
    // addWeighted( chopped1, alpha, chopped2, beta, 0.0, dst);
    return image1;
}
int mousepointsx;
vector<Point> mouse_points;
int mousepointsy = 0; 
Mat result;
void getMouseClicks(int event, int x, int y, int flags, void* userdata){
    // vector<Point> mouse_click;
    // for(int i = 0; i< 4; i++){
    int count = 0;
    if (event == EVENT_LBUTTONDOWN){ //when left button clicked//
        mouse_points.push_back(Point(x,y));
        // while(1){
        //     circle(result, Point(x, y), 4, Scalar(0, 0, 255), 2);
        //     count++;
        //     if (count == 4) break;
        // }
        // mousepointsx[n] = x;
        // mousepointsx[n+1] = y;
        mousepointsx = x;
        mousepointsy = y;
        cout << "Left click has been made, Position:(" << mousepointsx << "," << mousepointsy << ")" << endl;
      }
    //   cout << "Four points are selected" << endl;
    // }
    
    // return mouse_click;
}

void blendded(Mat image){
    cvtColor(image,image,COLOR_BGR2GRAY);
    cv::Mat gradientMask(image.size(), CV_8UC1, cv::Scalar(0));
    int halfWidth = image.cols / 2;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            gradientMask.at<uchar>(i, j) = (j * 255) / halfWidth;
        }
    }
    cv::Mat blendedImage;
    cv::multiply(image, gradientMask, blendedImage);
    cv::Mat invertedMask;
    cv::bitwise_not(gradientMask, invertedMask);
    cv::Mat blendedInvertedImage;
    cv::multiply(image, invertedMask, blendedInvertedImage);
    cv::Mat finalImage;
    double alpha = 0.5;
    double beta = 1 - alpha;
    cv::addWeighted(blendedImage, alpha, blendedInvertedImage, beta, 0, finalImage);
    cv::imshow("Final Image", invertedMask);
    cv::waitKey(0);
}

int main(){
    string path = "/home/yash/Documents/Computer_VIsion/CV_2Project/DanaOffice/";
    Mat img1 = imread(path + string("DSC_0309.JPG"));
    Mat img2 = imread(path + string("DSC_0310.JPG"));
    Mat space  = imread("space-cartoon-animated-background-footage-075133825_iconl.webp");
    imshow("space",space);
    Mat imggray1,imggray2;
    // cvtColor(img1, imggray1, COLOR_BGR2GRAY);
    // cvtColor(img2, imggray2, COLOR_BGR2GRAY);
    Image_Mosaicing p3(path);
    vector<Point> cor_img1,cor_img2;
    // vector<Point> cor_img1,cor_img2;
    cor_img1 = p3.harris_detector_for_img1(234);
    cor_img2 = p3.harris_detector_for_img2(250);
    // tie(cor_img1,cor_img2) = p3.perform_harris(145);
    // for (auto e : cor_img2){
    //     cout << e << endl;
    // }
    vector<pair<Point,Point>> corres;
    corres = p3.get_correspondences(cor_img1,cor_img2);
    cout << "done" << endl;
    // // cout << corres << endl;
    // p3.visualise_corress(corres);
    vector<Point> src,dst;
    vector<DMatch> matches;
    for (int i = 0; i < corres.size(); i++) {
        src.push_back(corres[i].first);
        dst.push_back(corres[i].second);
    }
    // std::vector<Point2f> points1, points2; // your corresponding points
    Mat best_h;
    best_h = p3.estimate_homography_ransac(src,dst);
    cout << best_h << endl;
    // vector<Point> c1,c2;
    // c1.push_back(Point(0,0));
    // c1.push_back(Point(img1.cols,0));
    // c1.push_back(Point(img1.cols,img1.rows));
    // c1.push_back(Point(0,img1.rows));
    // for (int i=0;i<4;i++){
    //     Point src_point = c1[i];
    //     Mat src = (Mat_<double>(3, 1) << src_point.x, src_point.y, 1);
    //     Mat pred_dst = best_h * src;
    //     pred_dst /= pred_dst.at<double>(2, 0);
    //     c2.push_back(Point(pred_dst.at<double>(0,0),pred_dst.at<double>(1,0)));
    // }
    // cout << "here" << endl;
    // int min_x = min(min(c2[0].x,c2[1].x),min(c2[2].x,c2[3].x));
    // int max_x = max(max(c2[0].x,c2[1].x),max(c2[2].x,c2[3].x));

    // int min_y = min(min(c2[0].y,c2[1].y),min(c2[2].y,c2[3].y));
    // int max_y = max(max(c2[0].y,c2[1].y),max(c2[2].y,c2[3].y));

    // int height = max_y - min_y;
    // int width = max_x - min_x;
    // cout << max_x << endl;
    // cout << max_y << endl;
    // Mat output(1000,1000,img1.type());
    // warpPerspective(img1,output,best_h.inv(),output.size());
    // img1.copyTo(output(Rect(c2[0].x - min_x, c2[0].y - min_y, img1.cols, img1.rows)));
    // cout << "her3" << endl;
//     141
// 701
// 26
// 380


    blendded(img1);
    Mat r2;
    warpPerspective(img1, result, best_h, Size(img1.cols + img2.cols, img1.rows));
    Mat roi(result, Rect(0, 0, img2.cols, img2.rows));    
    img2.copyTo(roi);
    // r2.copyTo(result);
    // rectangle(result,Point(510,0),Point(514,340),Scalar(0,0,255),2,LINE_8);
    cout << img1.size() << endl;
    namedWindow("Stitched image", WINDOW_NORMAL);
    cout << result.size() << endl;
    setMouseCallback("Stitched image", getMouseClicks, NULL);
    // cv::Rect roi = cv::selectROI("Select frame", image1, false, false);
    // int overlap_width = img1.cols - 510;
    // int overlap_height = img1.rows - 340;
    // int start_x = img2.cols - overlap_width;
    // int start_y = img2.rows - overlap_height;
    // for (int i = 0; i < overlap_height; i++) {
    //     for (int j = 0; j < overlap_width; j++) {
    //         float w1 = (float)(j) / overlap_width;
    //         float w2 = 512 - 188;
    //         result.at<Vec3b>(340 + i, 512 + j) = (w1 * img1.at<Vec3b>(340 + i, 512 + j)) + (w2 * img2.at<Vec3b>(start_y + i, 500));
    //         }
    // }
    // for(int i=0; i<result.rows; i++)
    // {
    //     for(int j =141; j<701; j++)
    //     {
    //         // if the pixel is black, copy the pixel from the second image
    //         if(result.at<Vec3b>(i, j) == Vec3b(0, 0, 0))
    //         {
    //             if(i < img2.rows && j < img2.cols)
    //                 result.at<Vec3b>(i, j) = img2.at<Vec3b>(i, j);
    //         }
    //         // if the pixel is not black, average the pixel with the pixel from the second image
    //         else
    //         {
    //             if(i < img2.rows && j < img2.cols)
    //             {
    //                 result.at<Vec3b>(i, j)[0] = (img1.at<Vec3b>(i, j)[0]*(j-img1.cols) + img2.at<Vec3b>(i, j)[0]*(j-img1.cols))/(j-img1.cols + j-img1.cols);
    //                 result.at<Vec3b>(i, j)[1] = (img1.at<Vec3b>(i, j)[1]*(j-img1.cols) + img2.at<Vec3b>(i, j)[1]*(j-img1.cols))/(j-img1.cols + j-img1.cols);
    //                 result.at<Vec3b>(i, j)[2] = (img1.at<Vec3b>(i, j)[2]*(j-img1.cols) + img2.at<Vec3b>(i, j)[2]*(j-img1.cols))/(j-img1.cols + j-img1.cols);
    //             }
    //         }
    //     }
    // }
    // 
    imshow("Stitched image", result);
    imwrite("Weighted_avg.jpg",result);

     for(int i=0; i<result.rows; i++)
    {
        for(int j =141; j<701; j++)
        {
            // if the pixel is black, copy the pixel from the second image
            if(result.at<Vec3b>(i, j) == Vec3b(0, 0, 0))
            {
                if(i < img2.rows && j < img2.cols)
                    result.at<Vec3b>(i, j) = img2.at<Vec3b>(i, j);
            }
            // if the pixel is not black, average the pixel with the pixel from the second image
            else
            {
                if(i < img2.rows && j < img2.cols)
                {
                    result.at<Vec3b>(i, j)[0] = (img2.at<Vec3b>(i, j)[0]*(0.95) + img1.at<Vec3b>(i, j)[0]*(0.05));
                    result.at<Vec3b>(i, j)[1] = (img2.at<Vec3b>(i, j)[1]*(0.95) + img1.at<Vec3b>(i, j)[1]*(0.05));
                    result.at<Vec3b>(i, j)[2] = (img2.at<Vec3b>(i, j)[2]*(0.95) + img1.at<Vec3b>(i, j)[2]*(0.05));
                }
            }
        }
    }

    imshow("Stitched image with feather avg", result);
    // p3.find_ovelapping_region(best_h);
    // Mat blended_img = p3.alpha_blending(img1,img2,0.4);
    // imshow("alpha blending",blended_img);
    // imwrite("alpha blending05.jpg",blended_img);

    // Mat blended_img2 = p3.alpha_blending(result,img2,0.9);
    // imshow("alpha blending2",blended_img2);
    // imwrite("alpha blending09.jpg",blended_img2);
    // cout << "outpus" << endl;
    
    
    // imshow("parital",img2(Rect(141, 0, 371, 340)));
    // imshow("parital1",img1(Rect(0, 0, 371, 340)));
    // Mat overlap;
    // threshold(img1,imggray1,127,255,0);
    // threshold(img2,imggray2,127,255,0);
    // bitwise_not(imggray2,imggray1,overlap);
    // imshow("overlap", overlap);
    // cv::imwrite("stiched image.jpg", result);
    waitKey(0);
    // destroyAllWindows();
    // return 0;
    // cout << "her4" << endl;
    // warpPerspective(img2, output, best_h, output.size());
    // // cout << src<< "    ";
    // cout << src.size() << endl;
    // cout << dst.size() << endl;
    // Mat result;
    // warpPerspective(img1, result, best_h, Size(img1.cols + img2.cols, img1.rows));
    // // warpPerspective(img2, result, best_h, Size(img1.cols + img2.cols, img1.rows));
    // Mat roi(result, Rect(0, 0, img2.cols, img2.rows));
    // img2.copyTo(roi);
    // namedWindow("Stitched image", WINDOW_NORMAL);
    // imshow("Stitched image", output);
    // waitKey(0);
    // destroyAllWindows();
    // cv::imwrite("stiched image.jpg", img1);

}