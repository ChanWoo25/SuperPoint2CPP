/**
 * File: BowVector.cpp
 * Date: March 2011
 * Author: Dorian Galvez-Lopez
 * Description: bag of words vector
 * License: see the LICENSE.txt file
 *
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "BowVector.h"

namespace DBoW2 {

// --------------------------------------------------------------------------

BowVector::BowVector(void)
{
}

// --------------------------------------------------------------------------

BowVector::~BowVector(void)
{
}

// --------------------------------------------------------------------------

/**
 * @brief WordId id가 이미 BowVector에 존재하면, v값을 덧셈. 
 * 존재하지 않으면, 새로 생성.
 * 
 * @param id 
 * @param v 
 */
void BowVector::addWeight(WordId id, WordValue v)
{
  BowVector::iterator vit = this->lower_bound(id);
  
  if(vit != this->end() && !(this->key_comp()(id, vit->first)))
  {
    vit->second += v;
  }
  else
  {
    this->insert(vit, BowVector::value_type(id, v));
  }
}

// --------------------------------------------------------------------------

/**
 * @brief BowVector는 map구조를 가지고 있다. 이 map에 id를 키로 하는 Value가 존재하지 않으면 insert한다. 
 * map이지만 this[id]로 존재를 찾지 않고, lower_bound함수를 이용하는 것이 인상깊었다. 더 빠르게 구현되어 있나?
 * 
 * @param id 
 * @param v 
 */
void BowVector::addIfNotExist(WordId id, WordValue v)
{
  BowVector::iterator vit = this->lower_bound(id);
  
  if(vit == this->end() || (this->key_comp()(id, vit->first)))
  {
    this->insert(vit, BowVector::value_type(id, v));
  }
}

// --------------------------------------------------------------------------

/**
 * @brief norm_type에 맞게 BowVector 전체를 normalize시킨다.
 * 
 * @param norm_type 
 */
void BowVector::normalize(LNorm norm_type)
{
  double norm = 0.0; 
  BowVector::iterator it;

  if(norm_type == DBoW2::L1)
  {
    for(it = begin(); it != end(); ++it)
      norm += fabs(it->second);
  }
  else
  {
    for(it = begin(); it != end(); ++it)
      norm += it->second * it->second;
		norm = sqrt(norm);  
  }

  if(norm > 0.0)
  {
    for(it = begin(); it != end(); ++it)
      it->second /= norm;
  }
}

// --------------------------------------------------------------------------

std::ostream& operator<< (std::ostream &out, const BowVector &v)
{
  BowVector::const_iterator vit;
  unsigned int i = 0; 
  const unsigned int N = v.size();
  for(vit = v.begin(); vit != v.end(); ++vit, ++i)
  {
    out << "<" << vit->first << ", " << vit->second << ">";
    
    if(i < N-1) out << ", ";
  }
  return out;
}

// --------------------------------------------------------------------------

void BowVector::saveM(const std::string &filename, size_t W) const
{
  std::fstream f(filename.c_str(), std::ios::out);
  
  WordId last = 0;
  BowVector::const_iterator bit;
  for(bit = this->begin(); bit != this->end(); ++bit)
  {
    for(; last < bit->first; ++last)
    {
      f << "0 ";
    }
    f << bit->second << " ";
    
    last = bit->first + 1;
  }
  for(; last < (WordId)W; ++last)
    f << "0 ";
  
  f.close();
}

// --------------------------------------------------------------------------

} // namespace DBoW2

