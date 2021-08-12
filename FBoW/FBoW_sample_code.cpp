#include "dbow2/TemplatedVocabulary.h"
#include "dbow2/FORB.h"

#include "fbow.h"
#include <iomanip>
#include <chrono>
#include <opencv2/flann.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <easy/profiler.h>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif
using ORBVocabulary=DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ;


vector< cv::Mat  >  loadFeatures(std::string imgPath, const int imgNum, string descriptor);


// Paths
const std::string vocPath = "../../vocabularies/ORBvoc.fbow";
const std::string databasePath = "../../Database/";
const std::string queryPath = "../../Query/";


// Number of Database Images
const int NDBIMAGES = 4;
const int NQUERYIMAGES = 1;

using namespace std;

int main(int argc,char**argv)
{
    // Initiate Profiler
    EASY_MAIN_THREAD;
    EASY_PROFILER_ENABLE;

    
    // Load Vocabulary
    EASY_BLOCK("Loading Vocabulary", profiler::colors::Lime);
    fbow::Vocabulary voc;
    cout << "Loading Vocabulary file... \n" << endl;
    voc.readFromFile(vocPath);
    EASY_END_BLOCK;


    
    // Extract Features from DB images
    EASY_BLOCK("Extract DB Features", profiler::colors::LightBlue);
    cout << "Extracting DB Features... \n" << endl;
    vector < vector< cv::Mat  > >dbFeatures(NDBIMAGES);
    for (int i = 0; i < NDBIMAGES; ++i)
    {
        stringstream ssDB;
        ssDB << databasePath << setfill('0') << setw(6) << i << ".png";
        dbFeatures[i] = loadFeatures(ssDB.str(), NDBIMAGES, "orb");
    }
    EASY_END_BLOCK;
    

    // Extract Features from QUERY images
    EASY_BLOCK("Extract QUERY Features", profiler::colors::LightGreen100);
    cout << "Extracting QUERY Features... \n" << endl;
    vector < vector< cv::Mat  > >queryFeatures(NQUERYIMAGES);
    for (int i = 0; i < NQUERYIMAGES; ++i)
    {
        stringstream ssQUERY;
        ssQUERY << queryPath << setfill('0') << setw(6) << i << ".png";
        queryFeatures[i] = loadFeatures(ssQUERY.str(), NQUERYIMAGES, "orb");
    }
    EASY_END_BLOCK;
        

    // Query Image
    EASY_BLOCK("Calculate Similarity", profiler::colors::Magenta);
    fbow::fBow vv, vv2;
    map<double, int, greater<double>> scores;
    for (size_t i = 0; i < queryFeatures.size(); i++)
    {
        vv = voc.transform(queryFeatures[i][0]);

        for (size_t j = 0; j < dbFeatures.size(); j++)
        {
            vv2 = voc.transform(dbFeatures[j][0]);
            double score = vv.score(vv, vv2);
            
            scores.insert(pair<double, int>(score, j));
            
            printf("%f, ", score);
        }
        printf("\n");
    }
    EASY_END_BLOCK;


    // Search Image by index and Save 
    EASY_BLOCK("Save Result Image", profiler::colors::Olive);
    for(int i = 0; i < NQUERYIMAGES; i++)
    {
        stringstream ssR;
        ssR << databasePath << setfill('0') << setw(6) << scores.begin()->second << ".png";
        cv::Mat image = cv::imread(ssR.str(), 0);

        stringstream ssW;
        ssW << queryPath << "Result" << i << ".png";
        cv::imwrite(ssW.str(), image);
    }
    EASY_END_BLOCK;

    profiler::dumpBlocksToFile("FBoW.prof");

    return 0;
}


vector< cv::Mat  >  loadFeatures(std::string imgPath, const int imgNum, string descriptor = "")  
{
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor == "orb")        fdetector = cv::ORB::create(2000);
    else if (descriptor == "brisk") fdetector = cv::BRISK::create();
#ifdef OPENCV_VERSION_3
    else if (descriptor == "akaze") fdetector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 3, 1e-4);
#endif
#ifdef USE_CONTRIB
    else if (descriptor == "surf")  fdetector = cv::xfeatures2d::SURF::create(15, 4, 2);
#endif

    else throw std::runtime_error("Invalid descriptor");
    assert(!descriptor.empty());
    vector<cv::Mat> features;
    

    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Mat image = cv::imread(imgPath, 0);
    if (image.empty())throw std::runtime_error("Could not open image" + imgPath);

    fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
    features.push_back(descriptors);
    
    return features;
}

