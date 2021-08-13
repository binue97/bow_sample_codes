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


void loadDBFeatures(DescriptorVector &vfeatures);
void loadVocabulary(OrbVocabulary &voc);
void changeStructure(const cv::Mat &plain, Descriptors &out);
void createDatabase(OrbDatabase &db, const DescriptorVector &vfeatures);
void loadQueryFeatures(DescriptorVector &vfeatures);
void queryDatabase(OrbDatabase &db, const DescriptorVector &vfeatures, std::vector<int> &imgIdx);
void saveResult(std::vector<int> &imgIdx);


// Number of Images to Build Database
const int NDBIMAGES = 4;
// Number of Images to Query with Database
const int NQUERYIMAGES = 1;


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

  loadDBFeatures(dbFeatures);

  loadVocabulary(*ptrVocabulary);
  
  EASY_BLOCK("Initialize Database", profiler::colors::LightBlue);
  OrbDatabase* ptrDatabase = new OrbDatabase(*ptrVocabulary, false, 0);
  EASY_END_BLOCK;

  createDatabase(*ptrDatabase, dbFeatures);

  loadQueryFeatures(queryFeatures);

  queryDatabase(*ptrDatabase, queryFeatures, imgIndex);

  saveResult(imgIndex);

  profiler::dumpBlocksToFile("DBoW2.prof");
  return 0;
}


void loadDBFeatures(DescriptorVector &vfeatures)
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
    cv::Mat mask;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    vfeatures.push_back(Descriptors());
    changeStructure(descriptors, vfeatures.back());
  }
}


void changeStructure(const cv::Mat &plain, Descriptors &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
    out[i] = plain.row(i);
  
}


void loadVocabulary(OrbVocabulary &voc)
{
  EASY_FUNCTION("Load Vocabulary", profiler::colors::Yellow);
  
  if(!(voc.loadFromTextFile(vocPath)))
  {
      std::cerr << "Wrong path to vocabulary. " << std::endl;
      std::cerr << "Falied to open at: " << vocPath << std::endl;
      exit(-1);
  }

  std::cout << "< Vocabulary information >" << std::endl
  << voc << std::endl << std::endl;

}


void createDatabase(OrbDatabase &db, const DescriptorVector &vfeatures)
{
  EASY_FUNCTION("Create Database", profiler::colors::Magenta);

  for(int i = 0; i < NDBIMAGES; i++)
    db.add(vfeatures[i]);

  std::cout << "< Database information >" << std::endl << db << std::endl << std::endl;

}


void loadQueryFeatures(DescriptorVector &vfeatures)
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
    cv::Mat mask;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    vfeatures.push_back(Descriptors());
    changeStructure(descriptors, vfeatures.back());
  }

}


void queryDatabase(OrbDatabase &db, const DescriptorVector &vfeatures, std::vector<int> &imgIdx)
{
  EASY_FUNCTION("Query Database", profiler::colors::DarkTeal);

  QueryResults ret;
  for(int i = 0; i < NQUERYIMAGES; i++)
  {
    // Query & Save 4 best match in QueryResult
    db.query(vfeatures[i], ret, 4);

    std::cout << "Searching for Image " << i << ". " << ret << std::endl << std::endl;

    // Save Image index of the Best Match for each Query Image
    imgIdx.push_back(ret[0].Id);
  }

}


void saveResult(std::vector<int> &imgIdx)
{
  // Search Image by index and Save 
  for(int i = 0; i < NQUERYIMAGES; i++)
  {
    std::stringstream ssR;
    ssR << dbPath << std::setfill('0') << std::setw(6) << imgIdx[i] << ".png";
    cv::Mat image = cv::imread(ssR.str(), 0);

    std::stringstream ssW;
    ssW << queryPath << "Result" << i << ".png";
    cv::imwrite(ssW.str(), image);
  }

}
