#include "dbow2/TemplatedVocabulary.h"
#include "dbow2/FORB.h"
#include "fbow.h"

#include <opencv2/flann.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <easy/profiler.h>

#include <iomanip>
#include <chrono>

using ORBVocabulary = DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ;
using DescriptorVector = std::vector<std::vector<cv::Mat>>; 

vector<cv::Mat> loadFeatures(const std::string &imgPath, int imgNum, const std::string &descriptor);


// Paths
const std::string vocPath = "../../vocabularies/ORBvoc.fbow";
const std::string databasePath = "../../Database/";
const std::string queryPath = "../../Query/";


// Number of Database Images
constexpr int NDBIMAGES = 4;
constexpr int NQUERYIMAGES = 1;


int main(int argc,char**argv)
{
    EASY_MAIN_THREAD;
    EASY_PROFILER_ENABLE;

    
    EASY_BLOCK("Loading Vocabulary", profiler::colors::Lime);
    fbow::Vocabulary voc;
    voc.readFromFile(vocPath);
    EASY_END_BLOCK;


    
    EASY_BLOCK("Extract DB Features", profiler::colors::LightBlue);
    DescriptorVector dbFeatures(NDBIMAGES);
    for (int i = 0; i < NDBIMAGES; ++i)
    {
        std::stringstream ssDB;
        ssDB << databasePath << setfill('0') << setw(6) << i << ".png";
        dbFeatures[i] = loadFeatures(ssDB.str(), NDBIMAGES, "orb");
    }
    EASY_END_BLOCK;
    

    EASY_BLOCK("Extract QUERY Features", profiler::colors::LightGreen100);
    DescriptorVector queryFeatures(NQUERYIMAGES);
    for (int i = 0; i < NQUERYIMAGES; ++i)
    {
        std::stringstream ssQUERY;
        ssQUERY << queryPath << setfill('0') << setw(6) << i << ".png";
        queryFeatures[i] = loadFeatures(ssQUERY.str(), NQUERYIMAGES, "orb");
    }
    EASY_END_BLOCK;
        

    // Query Image
    EASY_BLOCK("Calculate Similarity", profiler::colors::Magenta);
    fbow::fBow queryBoW, dbBoW;
    std::map<double, int, greater<double>> scores;
    for (uint64_t i = 0; i < queryFeatures.size(); i++)
    {
        queryBoW = voc.transform(queryFeatures[i][0]);

        for (size_t j = 0; j < dbFeatures.size(); j++)
        {
            dbBoW = voc.transform(dbFeatures[j][0]);
            double score = queryBoW.score(queryBoW, dbBoW);
            
            scores.insert(pair<double, int>(score, j));
            
            std::cout << score << std::endl;
        }
        std::cout << std::endl;
    }
    EASY_END_BLOCK;


    // Search Image by index and Save 
    EASY_BLOCK("Save Result Image", profiler::colors::Olive);
    for(int i = 0; i < NQUERYIMAGES; i++)
    {
        std::stringstream ssR;
        ssR << databasePath << setfill('0') << setw(6) << scores.begin()->second << ".png";
        cv::Mat image = cv::imread(ssR.str(), 0);

        std::stringstream ssW;
        ssW << queryPath << "Result" << i << ".png";
        cv::imwrite(ssW.str(), image);
    }
    EASY_END_BLOCK;

    profiler::dumpBlocksToFile("FBoW.prof");

    return 0;
}


vector<cv::Mat> loadFeatures(const std::string &imgPath, int imgNum, const std::string &descriptor = "")  
{
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor == "orb")        fdetector = cv::ORB::create(2000);

    else if (descriptor == "brisk") fdetector = cv::BRISK::create();

    else throw std::runtime_error("Invalid descriptor");
    assert(!descriptor.empty());
    std::vector<cv::Mat> features;
    
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Mat image = cv::imread(imgPath, 0);
    
    if (image.empty())
        throw std::runtime_error("Could not open image" + imgPath);

    fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
    features.push_back(descriptors);
    
    return features;
}

