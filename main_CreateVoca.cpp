/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <DBoW2.h>

using namespace DBoW2;
using namespace std;

/***************************************************************************/

void loadFeatures(vector<vector<cv::Mat > > &features);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void testVocCreation(const vector<vector<cv::Mat > > &features);
void testDatabase(const vector<vector<cv::Mat > > &features);

// number of training images
int N_IMAGE;

void wait()
{
  cout << endl << "Press enter to continue" << endl;
  getchar();
}

/***************************************************************************/

int main()
{
  vector<vector<cv::Mat > > features;
  N_IMAGE = 0;

  loadFeatures(features);

  testVocCreation(features);

  wait();

  testDatabase(features);

  return 0;
}

// ----------------------------------------------------------------------------

/**
 * @brief image에서 Feature를 뽑은 다음, vector<vector<TDescriptor>> 형태로 features에 저장.
 * 
 * @param features 
 */
void loadFeatures(vector<vector<cv::Mat > > &features)
{
    features.clear();
    features.reserve(N_IMAGE);

    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    cout << "Extracting ORB features..." << endl;
    for(int i = 0; i < N_IMAGE; ++i)
    {
        stringstream ss;
        ss << "images/image" << i << ".png";

        cv::Mat image = cv::imread(ss.str(), 0);
        cv::Mat mask;
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        orb->detectAndCompute(image, mask, keypoints, descriptors);

        features.push_back(vector<cv::Mat >());
        changeStructure(descriptors, features.back());
    }
}

// ----------------------------------------------------------------------------

/**
 * @brief 왜 굳이 과정을 분할하는지 모르겠지만... 
 * descriptor 벡터를 feature벡터에 집어넣기 위해 사용
 * 
 * @param plain input
 * @param out output
 */
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------

/**
 * @brief create the vocabulary
 * 
 * @param features loadfeature()를 통해 얻는 features 이중 벡터를 이용.
 */
void testVocCreation(const vector<vector<cv::Mat > > &features)
{
  // branching factor and depth levels 
  const int k = 9;
  const int L = 3;
  const WeightingType weight = TF_IDF;
  const ScoringType scoring = L1_NORM;

  SuperpointVocabulary voc(k, L, weight, scoring);

  cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
  /* Vocabulary 생성 함수 */
  voc.create(features);
  cout << "... done!" << endl;

  // cout을 이용한 vocabulary 정보 출력 가능.
  cout << "Vocabulary information: " << endl
  << voc << endl << endl;

  // lets do something with this vocabulary
  // voc를 클래스로 하여 feature정보를 BoWVector Type으로 변환하여 scoring 가능.
  // 기억을 되살리자면, BoWVector란, Vocabulary에 들어있는 word의 히스토그램을 얻고
  // 분별력을 더하기 위해 TF-IDF reweighting을 하여 얻은 벡터이다.
  cout << "Matching images against themselves (0 low, 1 high): " << endl;
  BowVector v1, v2;
  for(int i = 0; i < N_IMAGE; i++)
  {
    voc.transform(features[i], v1);
    for(int j = 0; j < N_IMAGE; j++)
    {
      voc.transform(features[j], v2);
      
      double score = voc.score(v1, v2);
      cout << "Image " << i << " vs Image " << j << ": " << score << endl;
    }
  }

  // save the vocabulary to disk
  cout << endl << "Saving vocabulary..." << endl;
  voc.save("small_voc.yml.gz");
  cout << "Done" << endl;
}

// ----------------------------------------------------------------------------

void testDatabase(const vector<vector<cv::Mat > > &features)
{
  cout << "Creating a small database..." << endl;

  // Load the vocabulary from disk
  SuperpointVocabulary voc("small_voc.yml.gz");
  
  SuperpointDatabase db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < N_IMAGE; i++)
  {
    db.add(features[i]);
  }

  cout << "... done!" << endl;

  cout << "Database information: " << endl << db << endl;

  // and query the database
  cout << "Querying the database: " << endl;

  // Vocabulary에 image에 대해 query. return type = QueryResults.
  QueryResults ret;
  for(int i = 0; i < N_IMAGE; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    // QueryResults 타입의 변수도 Cout으로 출력을 지원.
    cout << "Searching for Image " << i << ". " << ret << endl;
  }

  cout << endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  cout << "Saving database..." << endl;
  db.save("small_db.yml.gz");
  cout << "... done!" << endl;
  
  // once saved, we can load it again  
  cout << "Retrieving database once again..." << endl;
  SuperpointDatabase db2("small_db.yml.gz");
  cout << "... done! This is: " << endl << db2 << endl;
}

// ----------------------------------------------------------------------------


