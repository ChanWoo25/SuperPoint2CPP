/**
 * File: FSuperpoint.cpp
 * Date: September 2020
 * Author: Chanwoo Lee
 * Description: functions for SUPERPOINT descriptors
 * License: see the LICENSE.txt file
 *
 */
 
#include <vector>
#include <string>
#include <sstream>
#include <stdint.h>
#include <limits.h>

#include "FSuperpoint.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------
const int FSUPERPOINT::L = 256;

void FSUPERPOINT::meanValue(const std::vector<FSUPERPOINT::pDescriptor> &descriptors, 
  FSUPERPOINT::TDescriptor &mean)
{
  if(descriptors.empty())
  {
    return;
  }
  else if(descriptors.size() == 1)
  {
    mean = descriptors[0]->clone();
  }
  else
  {
    size_t len = descriptors.size();
    std::vector<DBoW2::FSUPERPOINT::pDescriptor>::const_iterator iter;
    for(iter = descriptors.begin(); iter != descriptors.end(); ++iter)
    {
        mean +=  *(*iter);
    }

    mean /= float(len);
  }
}

// --------------------------------------------------------------------------
  
double FSUPERPOINT::distance(const FSUPERPOINT::TDescriptor &a, 
  const FSUPERPOINT::TDescriptor &b)
{
  if(a.empty() || b.empty())
  {
    std::cout << "ERROR: Mat is empty!\n";
    exit(1);
  }

  TDescriptor dist = a - b;
  dist = dist * dist.t();

  double s = dist.at<float>(0);
  s /= FSUPERPOINT::L;

  return std::sqrt(s);
}

// --------------------------------------------------------------------------
  
std::string FSUPERPOINT::toString(const FSUPERPOINT::TDescriptor &a)
{
  stringstream ss;
  const float *ptr = a.ptr<float>(0);
  for(int i=0; i<FSUPERPOINT::L; i++){
      ss << *(ptr+i) << " ";
  }
  
  return ss.str();
}

// --------------------------------------------------------------------------
  
void FSUPERPOINT::fromString(FSUPERPOINT::TDescriptor &a, const std::string &s)
{
  a.create(1, FSUPERPOINT::L, CV_32F);
  float *ptr = a.ptr<float>(0);
  
  stringstream ss(s);
  for(int i = 0; i < FSUPERPOINT::L; ++i, ++ptr)
  {
    float n;
    ss >> n;
    
    if(!ss.fail()) 
      *ptr = (float)n;
  }
  
}

// --------------------------------------------------------------------------

void FSUPERPOINT::toMat32F(const std::vector<TDescriptor> &descriptors, 
  cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }
  const size_t N = descriptors.size();
  mat.create(N, FSUPERPOINT::L, CV_32F);
  cv::vconcat(descriptors, mat);
}

// --------------------------------------------------------------------------

void FSUPERPOINT::toMat32F(const cv::Mat &descriptors, cv::Mat &mat)
{
  descriptors.convertTo(mat, CV_32F);
}

// --------------------------------------------------------------------------

void FSUPERPOINT::toMat8U(const std::vector<TDescriptor> &descriptors, cv::Mat &mat)
{
    mat.create(descriptors.size(), FSUPERPOINT::L, CV_8U);
    mat.setTo(1);
}

// --------------------------------------------------------------------------

} // namespace DBoW2

