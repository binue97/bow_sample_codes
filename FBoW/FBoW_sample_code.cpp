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
#include <filesystem>
#include <string>

using ORBVocabulary = DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ;
using FeatureVector = std::vector<std::vector<cv::Mat>>; 
using FileLUT       = std::vector<std::string>;

namespace fs = std::filesystem;

vector<cv::Mat> loadFeatures(const std::string& imgPath, int imgNum, const std::string& descriptor);


// Paths
fs::path curPath = fs::current_path();
fs::path vocPath = curPath.parent_path().parent_path() / "vocabularies" / "ORBvoc.fbow";
fs::path databasePath = curPath.parent_path().parent_path() / "Database";
fs::path queryPath = curPath.parent_path().parent_path() / "Query";
fs::path savePath = curPath.parent_path().parent_path() / "Result" / "";


// Number of Database Images
constexpr int NDBIMAGES = 4;
// Number of Query Images
constexpr int NQUERYIMAGES = 1;


int main(int argc, char**argv)
{
    EASY_MAIN_THREAD;
    EASY_PROFILER_ENABLE;

    FileLUT dbTable;

    EASY_BLOCK("Loading Vocabulary", profiler::colors::Lime);
    fbow::Vocabulary voc;
    voc.readFromFile(vocPath);
    EASY_END_BLOCK;


    EASY_BLOCK("Extract DB Features", profiler::colors::LightBlue);
    FeatureVector dbFeatures(NDBIMAGES);
    int idx = 0;
    for(const auto& entry : fs::directory_iterator(databasePath))
    {
        const std::string fileName = entry.path();
        dbTable.push_back(fileName);
        dbFeatures[idx++] = loadFeatures(fileName, NDBIMAGES, "orb");
    }
    EASY_END_BLOCK;


    EASY_BLOCK("Extract QUERY Features", profiler::colors::LightGreen100);
    FeatureVector queryFeatures(NQUERYIMAGES);
    idx = 0;
    for(const auto& entry : fs::directory_iterator(queryPath))
    {
        const std::string fileName = entry.path();
        queryFeatures[idx++] = loadFeatures(fileName, NQUERYIMAGES, "orb");
    }
    EASY_END_BLOCK;
    

    // Query Image
    EASY_BLOCK("Calculate Similarity", profiler::colors::Magenta);
    fbow::fBow queryBoW, dbBoW;
    std::map<double, int, greater<double>> scores;
    for (uint64_t i = 0; i < queryFeatures.size(); i++)
    {
        queryBoW = voc.transform(queryFeatures[i][0]);

        for (uint64_t j = 0; j < dbFeatures.size(); j++)
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
        const std::string readPath = dbTable[scores.begin()->second];
        const cv::Mat image = cv::imread(readPath, 0);
        
        const std::string writePath = savePath.string() + "Result" + std::to_string(i) + ".png";
        cv::imwrite(writePath, image);
    }
    EASY_END_BLOCK;

    profiler::dumpBlocksToFile("FBoW.prof");

    return 0;
}


vector<cv::Mat> loadFeatures(const std::string& imgPath, int imgNum, const std::string& descriptor = "")  
{
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    assert(descriptor == "orb" || descriptor == "brisk");
    if (descriptor == "orb")        fdetector = cv::ORB::create(2000);

    else if (descriptor == "brisk") fdetector = cv::BRISK::create();

    else throw std::runtime_error("Invalid descriptor");

    std::vector<cv::Mat> features;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    const cv::Mat image = cv::imread(imgPath, 0);
    
    if (image.empty())
        throw std::runtime_error("Could not open image" + imgPath);

    fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
    features.push_back(descriptors);
    
    return features;
}

