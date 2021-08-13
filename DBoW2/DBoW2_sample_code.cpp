#include "DBoW2.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <easy/profiler.h>

#include <iostream>
#include <vector>
#include <iomanip>

using namespace DBoW2;
using DescriptorVector = std::vector<std::vector<cv::Mat>>;
using Descriptors = std::vector<cv::Mat>;


bool loadDBFeatures(DescriptorVector &vfeatures);
bool loadVocabulary(OrbVocabulary &voc);
void changeStructure(const cv::Mat &plain, Descriptors &out);
bool createDatabase(OrbDatabase &db, const DescriptorVector &vfeatures);
bool loadQueryFeatures(DescriptorVector &vfeatures);
bool queryDatabase(OrbDatabase &db, const DescriptorVector &vfeatures, std::vector<int> &imgIdx);
bool saveResult(std::vector<int> &imgIdx);


// Number of Images to Build Database
constexpr int NDBIMAGES = 4;
// Number of Images to Query with Database
constexpr int NQUERYIMAGES = 1;


// Paths 
const std::string vocPath = "../ORBvoc/ORBvoc.txt";
const std::string dbPath = "../demo/Database/";
const std::string queryPath = "../demo/Query/";


int main()
{
  EASY_MAIN_THREAD;
  EASY_PROFILER_ENABLE;

  // DataStructures
  std::vector<int> imgIndex;
  DescriptorVector dbFeatures;
  DescriptorVector queryFeatures;
  OrbVocabulary* ptrVocabulary = new OrbVocabulary();

  if(!loadDBFeatures(dbFeatures))
    std::cerr << "Error while loading DB features\n" << std::endl;

  if(!loadVocabulary(*ptrVocabulary))
    std::cerr << "Error while loading Vocabulary\n" << std::endl;
  
  EASY_BLOCK("Initialize Database", profiler::colors::LightBlue);
  OrbDatabase* ptrDatabase = new OrbDatabase(*ptrVocabulary, false, 0);
  EASY_END_BLOCK;

  if(!createDatabase(*ptrDatabase, dbFeatures))
    std::cerr << "Error while Creating DB\n" << std::endl;

  if(!loadQueryFeatures(queryFeatures))
    std::cerr << "Error loading Query features\n" << std::endl;

  if(!queryDatabase(*ptrDatabase, queryFeatures, imgIndex))
    std::cerr << "Error while Querying DB\n" << std::endl;

  if(!saveResult(imgIndex))
    std::cerr << "Error while saving Results\n" << std::endl; 

  profiler::dumpBlocksToFile("DBoW2.prof");
  return 0;
}


bool loadDBFeatures(DescriptorVector &vfeatures)
{
  EASY_FUNCTION("Load DB Descriptors", profiler::colors::LightGreen);

  vfeatures.clear();
  vfeatures.reserve(NDBIMAGES);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  // Feature Extraction for Database Images 
  for(int i = 0; i < NDBIMAGES; ++i)
  {
    std::stringstream ss;
    ss << dbPath << std::setfill('0') << std::setw(6) << i << ".png";

    cv::Mat image = cv::imread(ss.str(), 0);
    if(image.empty()) return false;
    cv::Mat mask;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    vfeatures.push_back(Descriptors());
    changeStructure(descriptors, vfeatures.back());
  }

  return true;
  
}


void changeStructure(const cv::Mat &plain, Descriptors &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
    out[i] = plain.row(i);
  
}


bool loadVocabulary(OrbVocabulary &voc)
{
  EASY_FUNCTION("Load Vocabulary", profiler::colors::Yellow);
  
  if(!voc.loadFromTextFile(vocPath))  return false;

  std::cout << "< Vocabulary information >" << std::endl
  << voc << std::endl << std::endl;

  return true;

}


bool createDatabase(OrbDatabase &db, const DescriptorVector &vfeatures)
{
  EASY_FUNCTION("Create Database", profiler::colors::Magenta);

  for(int i = 0; i < NDBIMAGES; i++)
    db.add(vfeatures[i]);

  std::cout << "< Database information >\n" << db << std::endl << std::endl;

  return true;
}


bool loadQueryFeatures(DescriptorVector &vfeatures)
{
  EASY_FUNCTION("Load Query Descriptors", profiler::colors::DeepOrange);

  vfeatures.clear();
  vfeatures.reserve(NQUERYIMAGES);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  // Feature Extraction for Query Images 
  for(int i = 0; i < NQUERYIMAGES; ++i)
  {
    std::stringstream ss;
    
    ss << queryPath << std::setfill('0') << std::setw(6) << i << ".png";

    cv::Mat image = cv::imread(ss.str(), 0);
    if(image.empty()) return false;
    cv::Mat mask;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    vfeatures.push_back(Descriptors());
    changeStructure(descriptors, vfeatures.back());
  }

  return true;

}


bool queryDatabase(OrbDatabase &db, const DescriptorVector &vfeatures, std::vector<int> &imgIdx)
{
  EASY_FUNCTION("Query Database", profiler::colors::DarkTeal);

  QueryResults ret;
  int candidates = 4;
  if(candidates > NQUERYIMAGES) return false;

  for(int i = 0; i < NQUERYIMAGES; i++)
  {
    // Query & Save 4 best match in QueryResult
    db.query(vfeatures[i], ret, candidates);

    std::cout << "Searching for Image " << i << ". " << ret << std::endl << std::endl;

    // Save Image index of the Best Match for each Query Image
    imgIdx.push_back(ret[0].Id);
  }

  return true;

}


bool saveResult(std::vector<int> &imgIdx)
{
  // Search Image by index and Save 
  for(int i = 0; i < NQUERYIMAGES; i++)
  {
    std::stringstream ssR;
    ssR << dbPath << std::setfill('0') << std::setw(6) << imgIdx[i] << ".png";
    cv::Mat image = cv::imread(ssR.str(), 0);
    if(image.empty()) return false;

    std::stringstream ssW;
    ssW << queryPath << "Result" << i << ".png";
    cv::imwrite(ssW.str(), image);
  }

  return true;

}
