/**
 * File: BowVector.h
 * Date: March 2011
 * Author: Dorian Galvez-Lopez
 * Description: bag of words vector
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_BOW_VECTOR__
#define __D_T_BOW_VECTOR__

#include <iostream>
#include <map>
#include <vector>

namespace DBoW2 {

/// Id of words
typedef unsigned int WordId;

/// Value of a word
typedef double WordValue;

/// Id of nodes in the vocabulary treee
typedef unsigned int NodeId;

/// L-norms for normalization
enum LNorm
{
  L1,
  L2
};

/// Weighting type
enum WeightingType
{
  TF_IDF,
  TF,
  IDF,
  BINARY
};

/// Scoring type
enum ScoringType
{
  L1_NORM,
  L2_NORM,
  CHI_SQUARE,
  KL,
  BHATTACHARYYA,
  DOT_PRODUCT
};

/// Vector of words to represent images

/**
 * @brief BowVector는 std::map을 상속받은 class로서 
 * WordId를 key로, WordValue를 value로 하여 map을 만든다.
 * 
 */
class BowVector: 
	public std::map<WordId, WordValue>
{
public:

	/** 
	 * Constructor
	 */
	BowVector(void);

	/**
	 * Destructor
	 */
	~BowVector(void);
	
	/**
	 * Adds a value to a word value existing in the vector, or creates a new
	 * word with the given value
	 * @param id word id to look for
	 * @param v value to create the word with, or to add to existing word
	 */
	void addWeight(WordId id, WordValue v);
	
	/**
	 * @brief BowVector는 map구조를 가지고 있다. 이 map에 id를 키로 하는 Value가 존재하지 않으면 insert한다. 
	 * .etc) map이지만 this[id]로 존재를 찾지 않고, lower_bound함수를 이용하는 것이 인상깊었다. 더 빠르게 구현되어 있나?
	 * 
	 * @param id 목표로 하는 word_id. 해당하는 value가 존재하지 않을 때만 v를 집어넣음
	 * @param v world value
	 */
	void addIfNotExist(WordId id, WordValue v);

	/**
	 * L1-Normalizes the values in the vector 
	 * @param norm_type norm used
	 */
	void normalize(LNorm norm_type);
	
	/**
	 * Prints the content of the bow vector
	 * @param out stream
	 * @param v
	 */
	friend std::ostream& operator<<(std::ostream &out, const BowVector &v);
	
	/**
	 * Saves the bow vector as a vector in a matlab file
	 * @param filename
	 * @param W number of words in the vocabulary
	 */
	void saveM(const std::string &filename, size_t W) const;
};

} // namespace DBoW2

#endif
