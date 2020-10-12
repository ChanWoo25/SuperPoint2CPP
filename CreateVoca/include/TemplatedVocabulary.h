/**
 * File: TemplatedVocabulary.h
 * Date: February 2011
 * Author: Dorian Galvez-Lopez
 * Description: templated vocabulary 
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_TEMPLATED_VOCABULARY__
#define __D_T_TEMPLATED_VOCABULARY__

#include <cassert>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <fstream>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "FeatureVector.h"
#include "FSuperpoint.h"
#include "BowVector.h"
#include "ScoringObject.h"

using namespace std;

namespace DBoW2 {

/// @param TDescriptor class of descriptor
/// @param F class of descriptor functions
template<class TDescriptor, class F>
/// Generic Vocabulary
class TemplatedVocabulary
{		
public:
	int MaxIterationPerCluster = 800;

	/**
	 * Initiates an empty vocabulary
	 * @param k branching factor
	 * @param L depth levels
	 * @param weighting weighting type
	 * @param scoring scoring type
	 */
	TemplatedVocabulary(int k = 10, int L = 5, 
		WeightingType weighting = TF_IDF, ScoringType scoring = L1_NORM);
	
	/**
	 * Creates the vocabulary by loading a file
	 * @param filename
	 */
	TemplatedVocabulary(const std::string &filename);
	
	/**
	 * Creates the vocabulary by loading a file
	 * @param filename
	 */
	TemplatedVocabulary(const char *filename);
	
	/** 
	 * Copy constructor
	 * @param voc
	 */
	TemplatedVocabulary(const TemplatedVocabulary<TDescriptor, F> &voc);
	
	/**
	 * Destructor
	 */
	virtual ~TemplatedVocabulary();
	
	/** 
	 * Assigns the given vocabulary to this by copying its data and removing
	 * all the data contained by this vocabulary before
	 * @param voc
	 * @return reference to this vocabulary
	 */
	TemplatedVocabulary<TDescriptor, F>& operator=(
		const TemplatedVocabulary<TDescriptor, F> &voc);
	
	/** 
	 * Creates a vocabulary from the training features with the already
	 * defined parameters
	 * @param training_features
	 */
	virtual void create
		(const std::vector<std::vector<TDescriptor> > &training_features);
	
	/**
	 * Creates a vocabulary from the training features, setting the branching
	 * factor and the depth levels of the tree
	 * @param training_features
	 * @param k branching factor
	 * @param L depth levels
	 */
	virtual void create
		(const std::vector<std::vector<TDescriptor> > &training_features, 
			int k, int L);

	/**
	 * Creates a vocabulary from the training features, setting the branching
	 * factor nad the depth levels of the tree, and the weighting and scoring
	 * schemes
	 */
	virtual void create
		(const std::vector<std::vector<TDescriptor> > &training_features,
			int k, int L, WeightingType weighting, ScoringType scoring);

	/**
	 * Returns the number of words in the vocabulary
	 * @return number of words
	 */
	virtual inline unsigned int size() const;
	
	/**
	 * Returns whether the vocabulary is empty (i.e. it has not been trained)
	 * @return true iff the vocabulary is empty
	 */
	virtual inline bool empty() const;

	virtual void transform(const std::vector<TDescriptor>& features, BowVector &v) 
		const;
	
	/**
	 * @brief 
	 * 
	 * @tparam TDescriptor 
	 * @tparam F 
	 * @param features input - image's descriptors
	 * @param v output - BowVector
	 * @param fv output - FeatureVector
	 * @param levelsup input - (m_L - levelsuppression)까지 탐색. 
	 */
	virtual void transform(const std::vector<TDescriptor>& features,
		BowVector &v, FeatureVector &fv, int levelsup) const;

	/**
	 * Transforms a single feature into a word (without weight)
	 * @param feature
	 * @return word id
	 */
	virtual WordId transform(const TDescriptor& feature) const;
	
	/**
	 * Returns the score of two vectors
	 * @param a vector
	 * @param b vector
	 * @return score between vectors
	 * @note the vectors must be already sorted and normalized if necessary
	 */
	inline double score(const BowVector &a, const BowVector &b) const;
	
	/**
	 * Returns the id of the node that is "levelsup" levels from the word given
	 * @param wid word id
	 * @param levelsup 0..L
	 * @return node id. if levelsup is 0, returns the node id associated to the
	 *   word id
	 */
	virtual NodeId getParentNode(WordId wid, int levelsup) const;
	
	/**
	 * Returns the ids of all the words that are under the given node id,
	 * by traversing any of the branches that goes down from the node
	 * @param nid starting node id
	 * @param words ids of words
	 */
	void getWordsFromNode(NodeId nid, std::vector<WordId> &words) const;
	
	/**
	 * Returns the branching factor of the tree (k)
	 * @return k
	 */
	inline int getBranchingFactor() const { return m_k; }
	
	/** 
	 * Returns the depth levels of the tree (L)
	 * @return L
	 */
	inline int getDepthLevels() const { return m_L; }
	
	/**
	 * Returns the real depth levels of the tree on average
	 * @return average of depth levels of leaves
	 */
	float getEffectiveLevels() const;
	
	/**
	 * Returns the descriptor of a word
	 * @param wid word id
	 * @return descriptor
	 */
	virtual inline TDescriptor getWord(WordId wid) const;
	
	/**
	 * Returns the weight of a word
	 * @param wid word id
	 * @return weight
	 */
	virtual inline WordValue getWordWeight(WordId wid) const;
	
	/** 
	 * Returns the weighting method
	 * @return weighting method
	 */
	inline WeightingType getWeightingType() const { return m_weighting; }
	
	/** 
	 * Returns the scoring method
	 * @return scoring method
	 */
	inline ScoringType getScoringType() const { return m_scoring; }
	
	/**
	 * Changes the weighting method
	 * @param type new weighting type
	 */
	inline void setWeightingType(WeightingType type);
	
	/**
	 * Changes the scoring method
	 * @param type new scoring type
	 */
	void setScoringType(ScoringType type);
		
	/**
	 * Loads the vocabulary from a text file
	 * @param filename
	 */
	bool loadFromTextFile(const std::string &filename);

	/**
	 * Saves the vocabulary into a text file
	 * @param filename
	 */
	void saveToTextFile(const std::string &filename) const; 

	/**
	 * Saves the vocabulary into a file
	 * @param filename
	 */
	void save(const std::string &filename) const;
	
	/**
	 * Loads the vocabulary from a file
	 * @param filename
	 */
	void load(const std::string &filename);
	
	/** 
	 * Saves the vocabulary to a file storage structure
	 * @param fn node in file storage
	 */
	virtual void save(cv::FileStorage &fs, 
		const std::string &name = "vocabulary") const;
	
	/**
	 * Loads the vocabulary from a file storage node
	 * @param fn first node
	 * @param subname name of the child node of fn where the tree is stored.
	 *   If not given, the fn node is used instead
	 */  
	virtual void load(const cv::FileStorage &fs, 
		const std::string &name = "vocabulary");
	
	/** 
	 * Stops those words whose weight is below minWeight.
	 * Words are stopped by setting their weight to 0. There are not returned
	 * later when transforming image features into vectors.
	 * Note that when using IDF or TF_IDF, the weight is the idf part, which
	 * is equivalent to -log(f), where f is the frequency of the word
	 * (f = Ni/N, Ni: number of training images where the word is present, 
	 * N: number of training images).
	 * Note that the old weight is forgotten, and subsequent calls to this 
	 * function with a lower minWeight have no effect.
	 * @return number of words stopped now
	 */
	virtual int stopWords(double minWeight);

protected:

	/// Pointer to descriptor
	typedef const TDescriptor *pDescriptor;

	/// Tree node
	struct Node 
	{
		/// Node id
		NodeId id;
		/// Weight if the node is a word
		WordValue weight;
		/// Children 
		std::vector<NodeId> children;
		/// Parent node (undefined in case of root)
		NodeId parent;
		/// Node descriptor
		TDescriptor descriptor;

		/// Word id if the node is a word
		WordId word_id;

		/**
		 * Empty constructor
		 */
		Node(): id(0), weight(0), parent(0), word_id(0){}
		
		/**
		 * Constructor
		 * @param _id node id
		 */
		Node(NodeId _id): id(_id), weight(0), parent(0), word_id(0){}

		/**
		 * Returns whether the node is a leaf node
		 * @return true iff the node is a leaf
		 */
		inline bool isLeaf() const { return children.empty(); }
	};

protected:

	/**
	 * Creates an instance of the scoring object accoring to m_scoring
	 */
	void createScoringObject();

	/** 
	 * Returns a set of pointers to descriptores
	 * @param training_features all the features
	 * @param features (out) pointers to the training features
	 */
	void getFeatures(
		const std::vector<std::vector<TDescriptor> > &training_features,
		std::vector<pDescriptor> &features) const;

	/**
	 * Returns the word id associated to a feature
	 * @param feature
	 * @param id (out) word id
	 * @param weight (out) word weight
	 * @param nid (out) if given, id of the node "levelsup" levels up
	 * @param levelsup
	 */
	virtual void transform(const TDescriptor &feature, 
		WordId &id, WordValue &weight, NodeId* nid = NULL, int levelsup = 0) const;

	/**
	 * Returns the word id associated to a feature
	 * @param feature
	 * @param id (out) word id
	 */
	virtual void transform(const TDescriptor &feature, WordId &id) const;
			
	/**
	 * Creates a level in the tree, under the parent, by running k-means with 
	 * a descriptor set, and recursively creates the subsequent levels, too.
	 * @param parent_id id of parent node
	 * @param descriptors descriptors to run the kmeans on
	 * @param current_level current level in the tree
	 */
	void HKmeansStep(NodeId parent_id, const std::vector<pDescriptor> &descriptors,
		int current_level);

	/**
	 * Creates k clusters from the given descriptors with some seeding algorithm.
	 * @note In this class, kmeans++ is used, but this function should be
	 *   overriden by inherited classes.
	 */
	virtual void initiateClusters(const std::vector<pDescriptor> &descriptors,
		std::vector<TDescriptor> &clusters) const;
	
	/**
	 * Creates k clusters from the given descriptor sets by running the
	 * initial step of kmeans++
	 * @param descriptors 
	 * @param clusters resulting clusters
	 */
	void initiateClustersKMpp(const std::vector<pDescriptor> &descriptors,
		std::vector<TDescriptor> &clusters) const;
	
	/**
	 * Create the words of the vocabulary once the tree has been built
	 */
	void createWords();
	
	/**
	 * Sets the weights of the nodes of tree according to the given features.
	 * Before calling this function, the nodes and the words must be already
	 * created (by calling HKmeansStep and createWords)
	 * @param features
	 */
	void setNodeWeights(const std::vector<std::vector<TDescriptor> > &features);
	
	/**
	 * Returns a random number in the range [min..max]
	 * @param min
	 * @param max
	 * @return random T number in [min..max]
	 */
	template <class T>
	static T RandomValue(T min, T max){
			return ((T)rand()/(T)RAND_MAX) * (max - min) + min;
	}

	/**
	 * Returns a random int in the range [min..max]
	 * @param min
	 * @param max
	 * @return random int in [min..max]
	 */
	static int RandomInt(int min, int max){
			int d = max - min + 1;
			return int(((double)rand()/((double)RAND_MAX + 1.0)) * d) + min;
	}

protected:

	/// Branching factor
	int m_k;
	
	/// Depth levels 
	int m_L;
	
	/// Weighting method
	WeightingType m_weighting;
	
	/// Scoring method
	ScoringType m_scoring;
	
	/// Object for computing scores
	GeneralScoring* m_scoring_object;
	
	/// Tree nodes
	std::vector<Node> m_nodes;
	
	/// Words of the vocabulary (tree leaves)
	/// this condition holds: m_words[wid]->word_id == wid
	std::vector<Node*> m_words;
	
};

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor,F>::TemplatedVocabulary
	(int k, int L, WeightingType weighting, ScoringType scoring)
	: m_k(k), m_L(L), m_weighting(weighting), m_scoring(scoring),
	m_scoring_object(NULL)
{
	createScoringObject();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor,F>::TemplatedVocabulary
	(const std::string &filename): m_scoring_object(NULL)
{
	load(filename);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor,F>::TemplatedVocabulary
	(const char *filename): m_scoring_object(NULL)
{
	load(filename);
}

// --------------------------------------------------------------------------

/**
 * @brief m_scoring에 따라 m_scoring_object를 초기화한다.
 * DotProductScoring을 제외하고는 대체적으로 normalize가 포함되어 있으며, 
 * L2 norm을 이용하는 것은 L2Scoring 밖에 없다.
 * @tparam TDescriptor 
 * @tparam F 
 */
template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::createScoringObject()
{
	delete m_scoring_object;
	m_scoring_object = NULL;
	
	switch(m_scoring)
	{
		case L1_NORM: 
			m_scoring_object = new L1Scoring;
			break;
			
		case L2_NORM:
			m_scoring_object = new L2Scoring;
			break;
		
		case CHI_SQUARE:
			m_scoring_object = new ChiSquareScoring;
			break;
			
		case KL:
			m_scoring_object = new KLScoring;
			break;
			
		case BHATTACHARYYA:
			m_scoring_object = new BhattacharyyaScoring;
			break;
			
		case DOT_PRODUCT:
			m_scoring_object = new DotProductScoring;
			break;
		
	}
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::setScoringType(ScoringType type)
{
	m_scoring = type;
	createScoringObject();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::setWeightingType(WeightingType type)
{
	this->m_weighting = type;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor,F>::TemplatedVocabulary(
	const TemplatedVocabulary<TDescriptor, F> &voc)
	: m_scoring_object(NULL)
{
	*this = voc;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor,F>::~TemplatedVocabulary()
{
	delete m_scoring_object;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedVocabulary<TDescriptor, F>& 
TemplatedVocabulary<TDescriptor,F>::operator=
	(const TemplatedVocabulary<TDescriptor, F> &voc)
{  
	this->m_k = voc.m_k;
	this->m_L = voc.m_L;
	this->m_scoring = voc.m_scoring;
	this->m_weighting = voc.m_weighting;

	this->createScoringObject();
	
	this->m_nodes.clear();
	this->m_words.clear();
	
	this->m_nodes = voc.m_nodes;
	this->createWords();
	
	return *this;
}

// --------------------------------------------------------------------------

/**
 * @brief Vocabulary 생성 함수. m_nodes, m_word가 비어있는 상태에서 시작. 
 * #1. feature pointer vector 생성(연산 효율 + 구조화) 
 * #2. HKmeansStep() 클러스터링 >> m_nodes생성 
 * #3. createWords() m_nodes가지고 m_words를 생성. 
 * #4. setNodeWeights() m_words의 모든 Node 원소의 Weight업데이트(IDF only)
 * 
 * @tparam TDescriptor 
 * @tparam F 
 * @param training_features 
 */
template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::create(
	const std::vector<std::vector<TDescriptor> > &training_features)
{
	m_nodes.clear();
	m_words.clear();
	std::cout << "===[Create]=== \n";
	// expected_nodes = Sum_{i=0..L} ( k^i )
	int expected_nodes = 
		(int)((pow((double)m_k, (double)m_L + 1) - 1)/(m_k - 1));

	m_nodes.reserve(expected_nodes); // avoid allocations when creating the tree
	
	std::vector<pDescriptor> features;
	getFeatures(training_features, features);
	// for(int i=0; i<5; i++)
	//   std::cout << *features[i] << std::endl; -> OK [-1,1] 사이의 값들로 잘 나옴. 

	// create root  
	m_nodes.push_back(Node(0)); // root
	
	// create the tree
	HKmeansStep(0, features, 1);

	// create the words
	createWords();

	// and set the weight of each node of the tree
	setNodeWeights(training_features);
	
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::create(
	const std::vector<std::vector<TDescriptor> > &training_features,
	int k, int L)
{
	m_k = k;
	m_L = L;
	
	create(training_features);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::create(
	const std::vector<std::vector<TDescriptor> > &training_features,
	int k, int L, WeightingType weighting, ScoringType scoring)
{
	m_k = k;
	m_L = L;
	m_weighting = weighting;
	m_scoring = scoring;
	createScoringObject();
	
	create(training_features);
}

// --------------------------------------------------------------------------

/**
 * @brief vector<vector<TDescriptor>> 형태로 가지고 있는 훈련 데이터를 변형할 일이 없기 때문에,
 * pointer만 싹 긁어서 vector<pDescriptor> 로 보관한다. 그 형태로 features로 반환해주는 함수.
 * 
 * @tparam TDescriptor 
 * @tparam F 
 * @param training_features Input : Detection&Description 과정을 거친 이미지당 feature를 가지고 있는 이중 벡터.
 * @param features Output : training_features의 각 elem의 Pointer를 일차원 벡터로 모아놓음
 */
template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::getFeatures(
	const std::vector<std::vector<TDescriptor> > &training_features,
	std::vector<pDescriptor> &features) const
{
	features.resize(0);
	
	typename std::vector<std::vector<TDescriptor> >::const_iterator vvit;
	typename std::vector<TDescriptor>::const_iterator vit;
	for(vvit = training_features.begin(); vvit != training_features.end(); ++vvit)
	{
		features.reserve(features.size() + vvit->size());   // size 늘림.
		for(vit = vvit->begin(); vit != vvit->end(); ++vit) // Descriptor's pointer를 한곳으로 몰아넣음
		{
			features.push_back(&(*vit));
		}
	}
	std::cout << "getFeatures() -- Get " << features.size() << "Features.\n\n";
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::HKmeansStep(NodeId parent_id, 
	const std::vector<pDescriptor> &descriptors, int current_level)
{
		if(descriptors.empty()) return;
		std::cout << "[HKmeansStep] [Lv. "<< current_level 
				  << "] [pFeatrues. " << descriptors.size() << "]\n";
		int cnt=0;

		// features associated to each cluster
		std::vector<TDescriptor> clusters;
		std::vector<std::vector<unsigned int> > groups;
		clusters.reserve(m_k);  // m_k 한 노드에서 가질 자식 개수.
		groups.reserve(m_k);    // cluster[i]가 k-means의 mean역할을 하고, 
														// 그 근처에 위치하는 descriptor들은 groups[i]에 인덱스로 저장. 
														// 이 인덱스는 descriptors vector의 인덱스 의미.
		
		//#############################################################################//

		if((int)descriptors.size() <= m_k)
		{
			// A. Trivial case: one cluster per feature
			std::cout << "A ";
			groups.resize(descriptors.size()); 

			for(unsigned int i = 0; i < descriptors.size(); i++)
			{
				groups[i].push_back(i);
				clusters.push_back(*(descriptors[i]));
			}	cnt++;
		}
		else
		{
			// B. Select clusters and groups with kmeans
			std::cout << "B ";
			bool first_time = true;
			bool go_on = true;
			
			// to check if clusters move after iterations
			std::vector<int> last_association, current_association;

			while(go_on)
			{
					/* 1. Calculate clusters */ 

					if(first_time)
					{
							// random sample 
							initiateClusters(descriptors, clusters);
							std::cout << "Let's calculate ...\n";
					}
					else
					{
							// calculate cluster centres
							std::cout << "="; cnt++;
							if(cnt % 100 == 0) cout << cnt << endl;
							
							for(unsigned int c = 0; c < clusters.size(); ++c)
							{
									std::vector<pDescriptor> cluster_descriptors;
									cluster_descriptors.reserve(groups[c].size());
									
									std::vector<unsigned int>::const_iterator vit;
									for(vit = groups[c].begin(); vit != groups[c].end(); ++vit)
									{
											cluster_descriptors.push_back(descriptors[*vit]);
									}
									
									// clusters[c]에 cluster_descriptors의 평균이 들어감.
									F::meanValue(cluster_descriptors, clusters[c]);
							}
					}

					/* 2. Associate features with clusters */ 

					// calculate distances to cluster centers
					groups.clear();
					groups.resize(m_k, std::vector<unsigned int>());
					current_association.resize(descriptors.size());

					// Group정보를 초기화하고 decriptors 각각을 가장 가까운 cluster에 할당.
					typename std::vector<pDescriptor>::const_iterator fit;
					for(fit = descriptors.begin(); fit != descriptors.end(); ++fit)
					{
						if((*fit)->empty())
						{
							std::cout << "fit empty\n";
						}
						else if(clusters[0].empty())
						{
								std::cout << "cluster[0] empty\n";
						}
						
						double best_dist = F::distance(*(*fit), clusters[0]);
						unsigned int icluster = 0;
						
						for(unsigned int c = 1; c < clusters.size(); ++c)
						{	
							if((*fit)->empty())
							{
								std::cout << "fit empty\n"; continue;
							}
							else if(clusters[c].empty())
							{
									std::cout << "cluster[" << c << "] empty\n"; continue;
							}

							double dist = F::distance(*(*fit), clusters[c]);
							if(dist < best_dist)
							{
									best_dist = dist;
									icluster = c;
							}
						}

						int fit_index = fit - descriptors.begin();
						groups[icluster].push_back(fit_index);
						current_association[fit_index] = icluster;
					}
					
					// kmeans++ ensures all the clusters has any feature associated with them

					// 3. check convergence
					if(first_time)
					{
							first_time = false;
					}
					else
					{
							go_on = false;
							for(unsigned int i = 0; i < current_association.size(); i++)
							{
									if((current_association[i] != last_association[i]) && cnt < MaxIterationPerCluster){
											go_on = true;
											break;
									}
							}
					}

					if(go_on)
					{
							// copy last feature-cluster association
							last_association = current_association;
					}
					
				} //END while(go_on)
			
		} //END else B.
		std::cout << "> " << cnt << "s try\n\n";
		
		// create nodes // m_nodes는 Vocabulary의 벡터.
		// m_k만큼 Node추가. 
		// m_nodes는 부모 자식을 같이 쭉 push_back으로 넣고 자식들이 parent_id를 가지고 있게 함.
		// m_nodes[i].children (Type) = vector<NodeID> 
		for(unsigned int i = 0; i < clusters.size(); ++i)
		{
				NodeId id = m_nodes.size(); //가장 끝 index를 가지게함.
				m_nodes.push_back(Node(id));
				m_nodes.back().descriptor = clusters[i];
				m_nodes.back().parent = parent_id;
				m_nodes[parent_id].children.push_back(id);
		}
		
		// go on with the next level
		if(current_level < m_L)
		{
				// iterate again with the resulting clusters
				const std::vector<NodeId> &children_ids = m_nodes[parent_id].children;
				for(unsigned int i = 0; i < clusters.size(); ++i)
				{
						NodeId id = children_ids[i];

						std::vector<pDescriptor> child_features;
						child_features.reserve(groups[i].size());

						std::vector<unsigned int>::const_iterator vit;
						for(vit = groups[i].begin(); vit != groups[i].end(); ++vit)
						{
								child_features.push_back(descriptors[*vit]);
						}

						if(child_features.size() > 1)
						{
								/* Recursive */
								HKmeansStep(id, child_features, current_level + 1);
						}
				}
		}
}

// --------------------------------------------------------------------------

/**
 * @brief Real implementation: initiateClustersKMpp -- 초기화 파트. 클러스터 없이 모여있는 descriptor 중 
 * 랜덤으로 하나를 골라 첫번째 center로 삼고 모든 descriptor와의 min_distance를 계산한다.
 * 그 후 center를 랜덤으로 추가해가면서, 모든 descriptor의 min_dist값을 업데이트하고
 * (조금 알 수 없는) 기준을 넘어가는 descriptor를 골라 또 센터로 삼고, 위 과정을 m_k번 반복한다.
 * 
 * @tparam TDescriptor - Descriptor Type
 * @tparam F - Function class
 * @param descriptors input Descriptors. 효율적 계산 위해, pointer vector로 활용.
 * @param clusters 코드 전체에 걸쳐 clusters[m_k]는 k 개의 cluster의 중심점 Descriptor역할을 함.
 */
template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor, F>::initiateClusters
	(const std::vector<pDescriptor> &descriptors,
	std::vector<TDescriptor> &clusters) const
{
	initiateClustersKMpp(descriptors, clusters);  
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::initiateClustersKMpp
	(const std::vector<pDescriptor> &pfeatures,
		std::vector<TDescriptor> &clusters) const
{
	// Implements kmeans++ seeding algorithm
	// Algorithm:
	// 1. Choose one center uniformly at random from among the data points.
	// 2. For each data point x, compute D(x), the distance between x and the nearest 
	//    center that has already been chosen.
	// 3. Add one new data point as a center. Each point x is chosen with probability 
	//    proportional to D(x)^2.
	// 4. Repeat Steps 2 and 3 until k centers have been chosen.
	// 5. Now that the initial centers have been chosen, proceed using standard k-means 
	//    clustering.
	std::cout << "[initiateClustersKMpp] => ";
	
	clusters.resize(0);
	clusters.reserve(m_k);

	// 각 descriptor별로 어떠한 center와의 거리든 가장 가까운 거리를 저장.
	std::vector<double> min_dists(pfeatures.size(), std::numeric_limits<double>::max());
	
	// 1. descriptor 중 랜덤하게 하나 선택.
	int ifeature = RandomInt(0, pfeatures.size() - 1);
	
	// create first cluster 
	//clusters[i]는 각 클러스터의 중심점의 역할을 한다. 여기서는 초기화이기 때문에 랜덤 선택하는것.
	clusters.push_back(*(pfeatures[ifeature]));

	// compute the initial distances
	typename std::vector<pDescriptor>::const_iterator fit;
	std::vector<double>::iterator dit;
	dit = min_dists.begin();
	for(fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)
	{
		*dit = F::distance(*(*fit), clusters.back());
	}  


	while((int)clusters.size() < m_k)
	{
		// 2. (1. 과정)을 반복. min_dist를 업데이트해준다는 것이 차이.
		dit = min_dists.begin();
		for(fit = pfeatures.begin(); fit != pfeatures.end(); ++fit, ++dit)
		{
			if(*dit > 0)
			{
				double dist = F::distance(*(*fit), clusters.back());
				if(dist < *dit) *dit = dist;
			}
		}
		
		// 3. 
		double dist_sum = std::accumulate(min_dists.begin(), min_dists.end(), 0.0);

		if(dist_sum > 0)
		{
			double cut_d;
			do
			{
				cut_d = RandomValue<double>(0, dist_sum);
			} while(cut_d == 0.0);

			double d_up_now = 0;
			for(dit = min_dists.begin(); dit != min_dists.end(); ++dit)
			{
				d_up_now += *dit;
				if(d_up_now >= cut_d) break;
			}
			
			if(dit == min_dists.end()) 
				ifeature = pfeatures.size()-1;
			else
				ifeature = dit - min_dists.begin();
			
			clusters.push_back(*pfeatures[ifeature]);

		} // if dist_sum > 0
		else
			break;
			
	} // while(used_clusters < m_k)
	std::cout << " DONE\n";
}

// --------------------------------------------------------------------------

/**
 * @brief HKmeansStep를 통해 완성된 m_nodes를 가지고 m_words를 생성하는 함수.
 * m_node는 parent와 children의 id로 구조가 짜여있긴하지만 우리가 사용하는건 
 * 가장 아래의 node, 즉 leaf node들이다. m_words는 m_nodes에서 children을 더이상 가지고 있지 않은 
 * Leaf node들의 pointer로 구성되어 있으며, m_words에 들어간 Node 타입 객체는 word_id를 부여받는다.
 * @tparam TDescriptor 
 * @tparam F 
 */
template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::createWords()
{
	m_words.resize(0);
	
	if(!m_nodes.empty())
	{
		m_words.reserve( (int)pow((double)m_k, (double)m_L) );

		typename std::vector<Node>::iterator nit;
		
		nit = m_nodes.begin(); // ignore root
		for(++nit; nit != m_nodes.end(); ++nit)
		{
			if(nit->isLeaf()) // children을 가지고 있지 않으면, true를 반환할 것으로 예상.
			{
				nit->word_id = m_words.size();  // Node.word_id를 
				m_words.push_back( &(*nit) );
			}
		}
	}
}

// --------------------------------------------------------------------------

/**
 * @brief m_words에 저장되어 있는 node는 (정확히는 node pointer) weight라는 변수를 
 * 멤버로 가진다. 이는 해당 Word가 얼마나 identity가 확실해서 image recognition에 도움을 주는지를 
 * 측정하여 가지는 점수라고 볼 수 있다. setNodeWeights() 는 training_features를 입력으로 받아 
 * m_words의 각 원소에 weight를 넣어주는 함수이다.
 * 
 * @tparam TDescriptor 
 * @tparam F 
 * @param training_features 
 */
template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::setNodeWeights
	(const std::vector<std::vector<TDescriptor> > &training_features)
{
	const unsigned int NWords = m_words.size();
	const unsigned int NDocs = training_features.size();

	if(m_weighting == TF || m_weighting == BINARY)
	{
		// idf part must be 1 always
		for(unsigned int i = 0; i < NWords; i++)
			m_words[i]->weight = 1;
	}
	else if(m_weighting == IDF || m_weighting == TF_IDF)
	{
		// IDF and TF-IDF: we calculte the idf path now

		// Note: this actually calculates the idf part of the tf-idf score.
		// The complete tf-idf score is calculated in ::transform

		// i-th word를 가지고 있는 image의 개수.
		std::vector<unsigned int> Ni(NWords, 0);  
		// 각 word가 체크되었는지 확인.
		std::vector<bool> counted(NWords, false); 
		
		typename std::vector<std::vector<TDescriptor> >::const_iterator mit;
		typename std::vector<TDescriptor>::const_iterator fit;

		// input으로 넣은 모든 descriptor에 대해서 word_id를 할당해줌.

		for(mit = training_features.begin(); mit != training_features.end(); ++mit)
		{
			fill(counted.begin(), counted.end(), false);

			for(fit = mit->begin(); fit < mit->end(); ++fit)
			{
				WordId word_id;
				// 아직 각 word의 weights가 초기화되있지 않은 상태이므로, word_id만을 추출.
				transform(*fit, word_id);

				if(!counted[word_id])
				{
					Ni[word_id]++; // mit가 가리키는 이미지에서 현재 word_id가 처음나왔으면 횟수 UP.
					counted[word_id] = true;
				}
			}
		}

		// 모든 word의 Weight업데이트! set ln(N/Ni)
		for(unsigned int i = 0; i < NWords; i++)
		{
			// TF-IDF가 아닌 IDF만이 구현되어 있음. 
			// kmeans++ 는 나타나지 않은 word는 m_word에 보관하지 않으므로 Ni[i] = 0인 경우는 나타나지 않음.
			if(Ni[i] > 0)
			{
				m_words[i]->weight = log((double)NDocs / (double)Ni[i]);
			}// else // This cannot occur if using kmeans++
		}
	
	}

}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
inline unsigned int TemplatedVocabulary<TDescriptor,F>::size() const
{
	return m_words.size();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
inline bool TemplatedVocabulary<TDescriptor,F>::empty() const
{
	return m_words.empty();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
float TemplatedVocabulary<TDescriptor,F>::getEffectiveLevels() const
{
	long sum = 0;
	typename std::vector<Node*>::const_iterator wit;
	for(wit = m_words.begin(); wit != m_words.end(); ++wit)
	{
		const Node *p = *wit;
		
		for(; p->id != 0; sum++) p = &m_nodes[p->parent];
	}
	
	return (float)((double)sum / (double)m_words.size());
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TDescriptor TemplatedVocabulary<TDescriptor,F>::getWord(WordId wid) const
{
	return m_words[wid]->descriptor;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
WordValue TemplatedVocabulary<TDescriptor, F>::getWordWeight(WordId wid) const
{
	return m_words[wid]->weight;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
WordId TemplatedVocabulary<TDescriptor, F>::transform
	(const TDescriptor& feature) const
{
	if(empty())
	{
		return 0;
	}
	
	WordId wid;
	transform(feature, wid);
	return wid;
}

// --------------------------------------------------------------------------

/**
 * @brief feature->Word 변환이 아닌 Image->BowVector변환을 수행.
 * 
 * @tparam TDescriptor 
 * @tparam F 
 * @param features 
 * @param v 
 */
template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::transform(
	const std::vector<TDescriptor>& features, BowVector &v) const
{
	v.clear();
	
	if(empty())
	{
		return;
	}

	// normalize 
	LNorm norm;
	// createScoringObject()에서 초기화된 m_scoring_object의 Score방식이 normalize가 필요할 경우 수행.
	bool must = m_scoring_object->mustNormalize(norm);

	typename std::vector<TDescriptor>::const_iterator fit;

	// We use TF_IDF.
	if(m_weighting == TF || m_weighting == TF_IDF)
	{
		for(fit = features.begin(); fit < features.end(); ++fit)
		{
			WordId id;
			WordValue w; 
			// w is the idf value if TF_IDF, 1 if TF
			
			transform(*fit, id, w);
			
			// not stopped
			// TF의 경우 word가 나온 횟수에 따라 가중치를 줘야하기 때문에 addWeight()를 사용한다.
			if(w > 0) v.addWeight(id, w);
		}
		
		// elementwise dividing by n_d. => n_id / n_d Complete
		if(!v.empty() && !must)
		{
			// unnecessary when normalizing
			const double nd = v.size();
			for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++) 
				vit->second /= nd;
		}
		
	}
	else // _IDF_ || _BINARY_
	{
		for(fit = features.begin(); fit < features.end(); ++fit)
		{
			WordId id;
			WordValue w;
			// w is idf if IDF, or 1 if BINARY
			
			transform(*fit, id, w);
			
			// not stopped
			// IDF에서는 word가 이미지에서 몇번이나 나왔는지는 중요치 않기 때문에 addIfNotExist함수 사용
			if(w > 0) v.addIfNotExist(id, w); 
			
		} // if add_features
	} // if m_weighting == ...
	
	if(must) v.normalize(norm);
}

// --------------------------------------------------------------------------

/**
 * @brief 
 * 
 * @tparam TDescriptor 
 * @tparam F 
 * @param features input - image's descriptors
 * @param v input&output - BowVector
 * @param fv output - FeatureVector
 * @param levelsup input - (m_L - levelsuppression)까지 탐색. 
 */
template<class TDescriptor, class F> 
void TemplatedVocabulary<TDescriptor,F>::transform(
	const std::vector<TDescriptor>& features,
	BowVector &v, FeatureVector &fv, int levelsup) const
{
	v.clear();
	fv.clear();
	
	if(empty()) // safe for subclasses
	{
		return;
	}
	
	// normalize 
	LNorm norm;
	bool must = m_scoring_object->mustNormalize(norm);
	
	typename std::vector<TDescriptor>::const_iterator fit;
	
	if(m_weighting == TF || m_weighting == TF_IDF)
	{
		unsigned int i_feature = 0;
		for(fit = features.begin(); fit < features.end(); ++fit, ++i_feature)
		{
			WordId id;
			NodeId nid;
			WordValue w; 
			// w is the idf value if TF_IDF, 1 if TF
			
			transform(*fit, id, w, &nid, levelsup);
			
			if(w > 0) // not stopped
			{ 
				v.addWeight(id, w);
				fv.addFeature(nid, i_feature);
			}
		}
		
		if(!v.empty() && !must)
		{
			// unnecessary when normalizing
			const double nd = v.size();
			for(BowVector::iterator vit = v.begin(); vit != v.end(); vit++) 
				vit->second /= nd;
		}
	
	}
	else // IDF || BINARY
	{
		unsigned int i_feature = 0;
		for(fit = features.begin(); fit < features.end(); ++fit, ++i_feature)
		{
			WordId id;
			NodeId nid;
			WordValue w;
			// w is idf if IDF, or 1 if BINARY
			
			transform(*fit, id, w, &nid, levelsup);
			
			if(w > 0) // not stopped
			{
				v.addIfNotExist(id, w);
				fv.addFeature(nid, i_feature);
			}
		}
	} // if m_weighting == ...
	
	if(must) v.normalize(norm);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F> 
inline double TemplatedVocabulary<TDescriptor,F>::score
	(const BowVector &v1, const BowVector &v2) const
{
	return m_scoring_object->score(v1, v2);
}

// --------------------------------------------------------------------------


template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::transform
	(const TDescriptor &feature, WordId &id) const
{
	WordValue weight;
	transform(feature, id, weight);
}

// --------------------------------------------------------------------------

/**
 * @brief 하나의 Descriptor를 가지고, 만들어져 있는 m_words에서 어디에 해당되는지 찾아 WordId를 반환.
 * 해당 word가 가지고 있는 weight도 가져올 수 있으며, 해당 Word가 m_nodes에서 어디에 위치하는지 알기 위해
 * NodeId도 얻을 수 있음.
 * [과정] level 1에서 클러스터링된 m_k개의 node와 distance비교: 가장 가까운 1개의 node 찾은 다음 
 * 한 level내려가서 level 2에 해당하는 자식들과 distance비교. 반복하여 (m_L - levelup)에 해당하는 level층까지 탐색. 
 * 만약 도중에 leafnode 이면 해당 노드에서 반환. 반환되는 중요 정보는 결국 word_id와 weight이다.
 * @tparam TDescriptor 
 * @tparam F 
 * @param feature input
 * @param id output
 * @param weight output - word의 weight 반환.
 * @param nid 반환하는 NodeId
 * @param levelup Node가 위치한 level을 특정하여 찾고 싶을 때, 값을 줄 수 있다.
 */
template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::transform(const TDescriptor &feature, 
	WordId &word_id, WordValue &weight, NodeId *nid, int levelsup) const
{ 
	// propagate the feature down the tree
	std::vector<NodeId> nodes;
	typename std::vector<NodeId>::const_iterator nit;

	// level at which the node must be stored in nid, if given
	const int nid_level = m_L - levelsup;
	if(nid_level <= 0 && nid != NULL) *nid = 0; // root

	NodeId final_id = 0; // root
	int current_level = 0;

	do
	{
		++current_level;
		nodes = m_nodes[final_id].children;
		final_id = nodes[0];

		double best_d = F::distance(feature, m_nodes[final_id].descriptor);

		for(nit = nodes.begin() + 1; nit != nodes.end(); ++nit)
		{
			NodeId id = *nit;
			double d = F::distance(feature, m_nodes[id].descriptor);
			if(d < best_d)
			{
				best_d = d;
				final_id = id;
			}
		}
		
		if(nid != NULL && current_level == nid_level)
			*nid = final_id;
		
	} while( !m_nodes[final_id].isLeaf() );

	// turn node id into word id
	word_id = m_nodes[final_id].word_id;
	weight = m_nodes[final_id].weight;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
NodeId TemplatedVocabulary<TDescriptor,F>::getParentNode
	(WordId wid, int levelsup) const
{
	NodeId ret = m_words[wid]->id; // node id
	while(levelsup > 0 && ret != 0) // ret == 0 --> root
	{
		--levelsup;
		ret = m_nodes[ret].parent;
	}
	return ret;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::getWordsFromNode
	(NodeId nid, std::vector<WordId> &words) const
{
	words.clear();
	
	if(m_nodes[nid].isLeaf())
	{
		words.push_back(m_nodes[nid].word_id);
	}
	else
	{
		words.reserve(m_k); // ^1, ^2, ...
		
		std::vector<NodeId> parents;
		parents.push_back(nid);
		
		while(!parents.empty())
		{
			NodeId parentid = parents.back();
			parents.pop_back();
			
			const std::vector<NodeId> &child_ids = m_nodes[parentid].children;
			std::vector<NodeId>::const_iterator cit;
			
			for(cit = child_ids.begin(); cit != child_ids.end(); ++cit)
			{
				const Node &child_node = m_nodes[*cit];
				
				if(child_node.isLeaf())
					words.push_back(child_node.word_id);
				else
					parents.push_back(*cit);
				
			} // for each child
		} // while !parents.empty
	}
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
int TemplatedVocabulary<TDescriptor,F>::stopWords(double minWeight)
{
	int c = 0;
	typename std::vector<Node*>::iterator wit;
	for(wit = m_words.begin(); wit != m_words.end(); ++wit)
	{
		if((*wit)->weight < minWeight)
		{
			++c;
			(*wit)->weight = 0;
		}
	}
	return c;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
bool TemplatedVocabulary<TDescriptor,F>::loadFromTextFile(const std::string &filename)
{
    ifstream f;
    f.open(filename.c_str());
	
    if(f.eof())
	return false;

    m_words.clear();
    m_nodes.clear();

    string s;
    getline(f,s);
    stringstream ss;
    ss << s;
    ss >> m_k;
    ss >> m_L;
    int n1, n2;
    ss >> n1;
    ss >> n2;

    if(m_k<0 || m_k>20 || m_L<1 || m_L>10 || n1<0 || n1>5 || n2<0 || n2>3)
    {
        std::cerr << "Vocabulary loading failure: This is not a correct text file!" << endl;
	return false;
    }
    
    m_scoring = (ScoringType)n1;
    m_weighting = (WeightingType)n2;
    createScoringObject();

    // nodes
    int expected_nodes =
    (int)((pow((double)m_k, (double)m_L + 1) - 1)/(m_k - 1));
    m_nodes.reserve(expected_nodes);

    m_words.reserve(pow((double)m_k, (double)m_L + 1));

    m_nodes.resize(1);
    m_nodes[0].id = 0;
    while(!f.eof())
    {
        string snode;
        getline(f,snode);
        stringstream ssnode;
        ssnode << snode;

        int nid = m_nodes.size();
        m_nodes.resize(m_nodes.size()+1);
	m_nodes[nid].id = nid;
	
        int pid ;
        ssnode >> pid;
        m_nodes[nid].parent = pid;
        m_nodes[pid].children.push_back(nid);

        int nIsLeaf;
        ssnode >> nIsLeaf;

        stringstream ssd;
        for(int iD=0;iD<F::L;iD++)
        {
            string sElement;
            ssnode >> sElement;
            ssd << sElement << " ";
	}
        F::fromString(m_nodes[nid].descriptor, ssd.str());

        ssnode >> m_nodes[nid].weight;

        if(nIsLeaf>0)
        {
            int wid = m_words.size();
            m_words.resize(wid+1);

            m_nodes[nid].word_id = wid;
            m_words[wid] = &m_nodes[nid];
        }
        else
        {
            m_nodes[nid].children.reserve(m_k);
        }
    }

    return true;

}

// --------------------------------------------------------------------------


template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::saveToTextFile(const std::string &filename) const
{
    fstream f;
    f.open(filename.c_str(),ios_base::out);
    f << m_k << " " << m_L << " " << " " << m_scoring << " " << m_weighting << endl;

    for(size_t i=1; i<m_nodes.size();i++)
    {
        const Node& node = m_nodes[i];

        f << node.parent << " ";
        if(node.isLeaf())
            f << 1 << " ";
        else
            f << 0 << " ";

        f << F::toString(node.descriptor) << " " << (double)node.weight << endl;
    }

    f.close();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::save(const std::string &filename) const
{
	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);
	if(!fs.isOpened()) throw std::string("Could not open file ") + filename;
	
	save(fs);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::load(const std::string &filename)
{
	cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);
	if(!fs.isOpened()) throw std::string("Could not open file ") + filename;
	
	this->load(fs);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::save(cv::FileStorage &f,
	const std::string &name) const
{
	// Format YAML:
	// vocabulary 
	// {
	//   k:
	//   L:
	//   scoringType:
	//   weightingType:
	//   nodes 
	//   [
	//     {
	//       nodeId:
	//       parentId:
	//       weight:
	//       descriptor: 
	//     }
	//   ]
	//   words
	//   [
	//     {
	//       wordId:
	//       nodeId:
	//     }
	//   ]
	// }
	//
	// The root node (index 0) is not included in the node vector
	//
	
	f << name << "{";
	
	f << "k" << m_k;
	f << "L" << m_L;
	f << "scoringType" << m_scoring;
	f << "weightingType" << m_weighting;
	
	// tree
	f << "nodes" << "[";
	std::vector<NodeId> parents, children;
	std::vector<NodeId>::const_iterator pit;

	parents.push_back(0); // root

	while(!parents.empty())
	{
		NodeId pid = parents.back();
		parents.pop_back();

		const Node& parent = m_nodes[pid];
		children = parent.children;

		for(pit = children.begin(); pit != children.end(); pit++)
		{
			const Node& child = m_nodes[*pit];

			// save node data
			f << "{:";
			f << "nodeId" << (int)child.id;
			f << "parentId" << (int)pid;
			f << "weight" << (double)child.weight;
			f << "descriptor" << F::toString(child.descriptor);
			f << "}";
			
			// add to parent list
			if(!child.isLeaf())
			{
				parents.push_back(*pit);
			}
		}
	}
	
	f << "]"; // nodes

	// words
	f << "words" << "[";
	
	typename std::vector<Node*>::const_iterator wit;
	for(wit = m_words.begin(); wit != m_words.end(); wit++)
	{
		WordId id = wit - m_words.begin();
		f << "{:";
		f << "wordId" << (int)id;
		f << "nodeId" << (int)(*wit)->id;
		f << "}";
	}
	
	f << "]"; // words

	f << "}";

}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedVocabulary<TDescriptor,F>::load(const cv::FileStorage &fs,
	const std::string &name)
{
	m_words.clear();
	m_nodes.clear();
	
	cv::FileNode fvoc = fs[name];
	
	m_k = (int)fvoc["k"];
	m_L = (int)fvoc["L"];
	m_scoring = (ScoringType)((int)fvoc["scoringType"]);
	m_weighting = (WeightingType)((int)fvoc["weightingType"]);
	
	createScoringObject();

	// nodes
	cv::FileNode fn = fvoc["nodes"];

	m_nodes.resize(fn.size() + 1); // +1 to include root
	m_nodes[0].id = 0;

	for(unsigned int i = 0; i < fn.size(); ++i)
	{
		NodeId nid = (int)fn[i]["nodeId"];
		NodeId pid = (int)fn[i]["parentId"];
		WordValue weight = (WordValue)fn[i]["weight"];
		std::string d = (std::string)fn[i]["descriptor"];
		
		m_nodes[nid].id = nid;
		m_nodes[nid].parent = pid;
		m_nodes[nid].weight = weight;
		m_nodes[pid].children.push_back(nid);
		
		F::fromString(m_nodes[nid].descriptor, d);
	}
	
	// words
	fn = fvoc["words"];
	
	m_words.resize(fn.size());

	for(unsigned int i = 0; i < fn.size(); ++i)
	{
		NodeId wid = (int)fn[i]["wordId"];
		NodeId nid = (int)fn[i]["nodeId"];
		
		m_nodes[nid].word_id = wid;
		m_words[wid] = &m_nodes[nid];
	}
}

// --------------------------------------------------------------------------

/**
 * Writes printable information of the vocabulary
 * @param os stream to write to
 * @param voc
 */
template<class TDescriptor, class F>
std::ostream& operator<<(std::ostream &os, 
	const TemplatedVocabulary<TDescriptor,F> &voc)
{
	os << "Vocabulary: k = " << voc.getBranchingFactor() 
		<< ", L = " << voc.getDepthLevels()
		<< ", Weighting = ";

	switch(voc.getWeightingType())
	{
		case TF_IDF: os << "tf-idf"; break;
		case TF: os << "tf"; break;
		case IDF: os << "idf"; break;
		case BINARY: os << "binary"; break;
	}

	os << ", Scoring = ";
	switch(voc.getScoringType())
	{
		case L1_NORM: os << "L1-norm"; break;
		case L2_NORM: os << "L2-norm"; break;
		case CHI_SQUARE: os << "Chi square distance"; break;
		case KL: os << "KL-divergence"; break;
		case BHATTACHARYYA: os << "Bhattacharyya coefficient"; break;
		case DOT_PRODUCT: os << "Dot product"; break;
	}
	
	os << ", Number of words = " << voc.size();

	return os;
}

} // namespace DBoW2

#endif
