#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <cstdint>

using namespace std;

void writeMatToFile(cv::Mat &m, const char *filename)
{
  ofstream fout(filename);

  if (!fout)
  {
    cout << "File Not Opened" << endl;
    return;
  }

  for (int i = 0; i < m.rows; i++)
  {
    for (int j = 0; j < m.cols; j++)
    {
      fout << m.at<double>(i, j) << "\t";
    }
    fout << endl;
  }

  fout.close();
}
cv::Mat GetInitialization(const cv::Mat &conf_disp)
{
  cv::Mat initialization = conf_disp.clone();

  int h = conf_disp.rows;
  int w = conf_disp.cols;
  double last_known = -1;
  double first_known = -1;

  for (int y = 0; y < h; y++)
  {
    for (int x = 0; x < w; x++)
    {
      if (conf_disp.at<double>(y, x) > 0)
      {
        last_known = conf_disp.at<double>(y, x);
      }
      if (first_known < 0)
      {
        first_known = last_known;
      }
      initialization.at<double>(y, x) = last_known;
    }
  }

  cv::Mat first_known_mat =
      cv::Mat::ones(h, w, initialization.type()) * first_known;
  cv::Mat mask = initialization < 0;
  first_known_mat.copyTo(initialization, mask);

  return initialization;
}

cv::Mat DensifyFrame(const cv::Mat &conf_disp, const cv::Mat &conf_map_64, const cv::Mat &hard_edges,
                     double lambda_d, double lambda_s, int num_solver_iterations, cv::Mat &initialization)
{
  int w = conf_disp.cols;
  int h = conf_disp.rows;
  int num_pixels = w * h;

  Eigen::SparseMatrix<double> A(num_pixels * 3, num_pixels);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(num_pixels * 3);
  Eigen::VectorXd x0 = Eigen::VectorXd::Zero(num_pixels);
  int num_entries = 0;

  initialization = GetInitialization(conf_disp);

  std::vector<Eigen::Triplet<double>> tripletList;

  for (int y = 1; y < h - 1; y++)
  {
    for (int x = 1; x < w - 1; x++)
    {
      int idx = x + y * w;
      x0(idx) = initialization.at<double>(y, x);
      if (conf_disp.at<double>(y, x) > 0.00)
      {
       double conf_weight = conf_map_64.at<double>(y,x);
        tripletList.emplace_back(
            Eigen::Triplet<double>(num_entries, idx, lambda_d*conf_weight));
        b(num_entries) = conf_disp.at<double>(y, x) * lambda_d*conf_weight;
        num_entries++;
      }

      if (hard_edges.at<uint8_t>(y, x) < 1)
      {
      if (hard_edges.at<uint8_t>(y, x) == hard_edges.at<uint8_t>(y - 1, x))
      {

        tripletList.emplace_back(
            Eigen::Triplet<double>(num_entries, idx - w, lambda_s));
        tripletList.emplace_back(
            Eigen::Triplet<double>(num_entries, idx, -lambda_s));
        b(num_entries) = 0;
        num_entries++;
      }

      if (hard_edges.at<uint8_t>(y, x) == hard_edges.at<uint8_t>(y, x - 1))
      {
        tripletList.emplace_back(
            Eigen::Triplet<double>(num_entries, idx - 1, lambda_s));
        tripletList.emplace_back(
            Eigen::Triplet<double>(num_entries, idx, -lambda_s));
        b(num_entries) = 0;
        num_entries++;
      }
      }
    }
  }

  A.setFromTriplets(tripletList.begin(), tripletList.end());

  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,
                           Eigen::Lower | Eigen::Upper>
      cg;

  cg.compute(A.transpose() * A);
  cg.setMaxIterations(num_solver_iterations);
  cg.setTolerance(1e-05);
  Eigen::VectorXd x_vec = cg.solveWithGuess(A.transpose() * b, x0);

  cv::Mat depth = cv::Mat::zeros(h, w, CV_64FC1);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      depth.at<double>(y, x) = x_vec(x + y * w);

  return depth;
}

int main(int argc, char *argv[])
{
  std::vector<std::string> args;
  args.assign(argv + 1, argv + argc);
  std::string conf_disp_path = args[0];
  std::string conf_map_path = args[1];
  std::string edge_path = args[2];
  double lambda_d = std::stod(args[3]);
  double lambda_s = std::stod(args[4]);
  int num_solver_iterations = std::stoi(args[5]);
  std::string inpainted_disp_path = args[6];
  int path_size = inpainted_disp_path.size();
  std::string refined_disp_path = inpainted_disp_path.substr(0, path_size - 4);
  refined_disp_path += "_refined.txt";
  std::string init_disp_path = inpainted_disp_path.substr(0, path_size-4);
  init_disp_path += "_init.txt";

  cv::Mat conf_disp = cv::imread(conf_disp_path, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
  double min, max;
  cv::minMaxLoc(conf_disp, &min, &max);
//  std::cout << "min: " << min << " max: " << max << std::endl;
  cv::Mat conf_disp_64;
  conf_disp.convertTo(conf_disp_64, CV_64F);
  conf_disp_64 = conf_disp_64 / 500.0;
  //    cv::minMaxLoc(conf_disp_64, &min, &max);
  //    std::cout << "min: " << min << " max: " << max << std::endl;
  //    writeMatToFile(conf_disp_64, "conf_disp_64.txt");

  cv::Mat conf_map = cv::imread(conf_map_path, 0);
  cv::Mat conf_map_64;
  conf_map.convertTo(conf_map_64, CV_64F);
  conf_map_64 = conf_map_64 / 255.0;
  cv::Mat edge = cv::imread(edge_path, 0);
//  cv::Mat edge_64;
//  edge.convertTo(edge_64, CV_64F);

  cv::Mat init_disp;
  cv::Mat depth_AR = DensifyFrame(conf_disp_64, conf_map_64, edge, lambda_d, lambda_s, num_solver_iterations, init_disp);

  writeMatToFile(init_disp, init_disp_path.c_str());
  writeMatToFile(depth_AR, inpainted_disp_path.c_str());

  int h = conf_disp_64.rows;
  int w = conf_disp_64.cols;
  for (int y = 0; y < h; y++)
  {
    for (int x = 0; x < w; x++)
    {
      if ((conf_disp_64.at<double>(y, x) - depth_AR.at<double>(y,x))> 2)
      {
        conf_disp_64.at<double>(y, x) = 0;
      }
      if ((conf_disp_64.at<double>(y, x) - depth_AR.at<double>(y,x)) < -2)
      {
        conf_disp_64.at<double>(y, x) = 0;
      }
    }
  }

  cv::Mat depth_refined = DensifyFrame(conf_disp_64, conf_map_64, edge, lambda_d, lambda_s, num_solver_iterations, init_disp);
  writeMatToFile(depth_refined, refined_disp_path.c_str());

}