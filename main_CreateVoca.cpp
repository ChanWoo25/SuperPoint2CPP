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
#include <opencv2/opencv.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/features2d.hpp>
#include <DBoW2.h>
#include <SPDetector.hpp>
#include <Tools.hpp>

using namespace DBoW2;
using namespace std;
using namespace SuperPointSLAM;

/**  You need to modify the path below that corresponds to your dataset and weight path. **/
const std::string project_dir = "/home/leecw/Reps/SuperPoint2CPP/";
const std::string weight_dir = project_dir + "Weights/superpoint.pt";
const std::string dataset_dir = project_dir + "Dataset/";
/***************************************************************************/

void SuperpointVocCreation(const vector<vector<cv::Mat > > &features);
void TestDatabase(const vector<vector<cv::Mat > > &features);

// number of training images
int N_IMAGE = 5;

void wait()
{
    cout << endl << "Press enter to continue" << endl;
    getchar();
}

void test()
{

    float a[] = {1, 2, 3, 4, 5, 6};
    float b[] = {4, 5, 6};
    cv::Mat A(3, 2, CV_32F, a);
    cv::Mat B(3, 1, CV_32F, b);
    std::cout << A << std::endl;
    std::cout << A.row(0) << std::endl;
    std::cout << A.row(1) << std::endl;
    std::cout << A.col(0) << std::endl;
    std::cout << A.col(0) << std::endl;
    std::cout << A.size() << std::endl;
    double c = cv::sum(A)[0];
    double d = A.at<float>(0, 0);

    std::cout << c << std::endl;
    std::cout << std::sqrt(c) << std::endl;

    // std::cout << d << std::endl;
    // std::cout << A.at<float>(0) << std::endl;
    // std::cout << B.at<float>(0) << std::endl;
}

/***************************************************************************/

int main(int argc, char* argv[])
{
    cv::String DATA_PATH;
    if(argc == 2)
    {
        DATA_PATH = cv::String(argv[1]);
        cout << DATA_PATH << endl;
    }


    // test();


    vector< vector<cv::Mat> > features;
    VideoStreamer vs("/home/leecw/Datasets/Soongsil_Post/SoongsilMixed%4d.png");

    /** Superpoint Detector **/
    SPDetector SPF(weight_dir, torch::cuda::is_available());
    std::cout << "VC created, SPDetector Constructed.\n";

    long long cnt = 0;
    int t = N_IMAGE;
    while(t--){
        if(!vs.next_frame()) 
        { 
            std::cout << "main -- Image End\n"; 
            break; 
        }

        features.push_back(vector<cv::Mat>());
        features[cnt].resize(0);

        cv::Mat* descriptors = SPF.detect(vs.input);  // [N_kpts, 256]  Size format:[W, H]
        int len = descriptors->size().height;

        for(unsigned i = 0; i < len; i++)
        {
            //std::cout << "======================< " << i << " >======================\n"; 
            //std::cout << descriptors->row(i) << std::endl;
            features[cnt].push_back(descriptors->row(i));
            //getchar();
        }
        

        // for(unsigned i = 0; i<3; i++)
        // {   
        //     auto m = features[cnt][i];
        //     std::cout << m.isContinuous() << std::endl;
        //     std::cout << m.size() << std::endl;
        //     wait();
        // }
        

        cnt++;  
    }

    SuperpointVocCreation(features);

    wait();

    TestDatabase(features);

    return 0;
}

// ----------------------------------------------------------------------------

/**
 * @brief create the vocabulary
 * 
 * @param features loadfeature()를 통해 얻는 features 이중 벡터를 이용.
 */
void SuperpointVocCreation(const vector<vector<cv::Mat > > &features)
{
    // branching factor and depth levels 
    const int k = 6;
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

void TestDatabase(const vector<vector<cv::Mat > > &features)
{
    cout << "Creating a small database..." << endl;

    // Load the vocabulary from disk
    SuperpointVocabulary voc("small_voc.yml.gz");
    
    SuperpointDatabase db(voc, false, 0); 
    // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that 
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, 
    // we may get rid of "voc" now

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
    for(int i = 0; i < 5; i++)
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


