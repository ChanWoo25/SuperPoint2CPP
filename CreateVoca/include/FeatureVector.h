/**
 * File: FeatureVector.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: feature vector
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_FEATURE_VECTOR__
#define __D_T_FEATURE_VECTOR__

#include "BowVector.h"
#include <map>
#include <vector>
#include <iostream>

namespace DBoW2 {

/**
 * @brief Vector of nodes with indexes of local features 
 * 하나의 이미지에 들어있는 feature들이 있을 때의 인덱스를 이용하여, <NodeId, Local Index>
 * 를 가지는 map구조. 
 * FeatureVector는 이미지당 하나의 객체를 가질 수 있으며, 어떤 NodeId에 어떠어떠한 feature가 포함되어 있는지 담는다.
 * 
 */
class FeatureVector: 
  public std::map<NodeId, std::vector<unsigned int> >
{
public:

  /**
   * Constructor
   */
  FeatureVector(void);
  
  /**
   * Destructor
   */
  ~FeatureVector(void);
  
  /**
   * Adds a feature to an existing node, or adds a new node with an initial
   * feature
   * @param id node id to add or to modify
   * @param i_feature index of feature to add to the given node
   */
  void addFeature(NodeId id, unsigned int i_feature);

  /**
   * Sends a string versions of the feature vector through the stream
   * @param out stream
   * @param v feature vector
   */
  friend std::ostream& operator<<(std::ostream &out, const FeatureVector &v);
    
};

} // namespace DBoW2

#endif

