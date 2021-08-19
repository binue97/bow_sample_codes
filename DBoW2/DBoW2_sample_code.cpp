#include "DBoW2.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <easy/profiler.h>

#include <iostream>
#include <vector>
#include <filesystem>
#include <string>
#include <memory>

using namespace DBoW2;
namespace fs = std::filesystem;
namespace SampleCode
{
  using FeatureVector = std::vector<std::vector<cv::Mat>>;
  using Features = std::vector<cv::Mat>;
  using FileLUT = std::vector<std::string>;
}

bool loadDBFeatures(SampleCode::FeatureVector& vfeatures, SampleCode::FileLUT& dbTable, const cv::Ptr<cv::ORB>& orb);
bool loadVocabulary(OrbVocabulary& voc);
void changeStructure(const cv::Mat& plain, SampleCode::Features& out);
bool createDatabase(OrbDatabase& db, const SampleCode::FeatureVector& vfeatures);
bool loadQueryFeatures(SampleCode::FeatureVector& vfeatures, const cv::Ptr<cv::ORB>& orb);
bool queryDatabase(OrbDatabase& db, const SampleCode::FeatureVector& vfeatures, SampleCode::FileLUT& queryResultTable, const SampleCode::FileLUT& dbTable);
bool saveResult(const SampleCode::FileLUT& queryResultTable);


// Number of Images to Build Database
constexpr int NDBIMAGES = 4;
// Number of Images to Query with Database
constexpr int NQUERYIMAGES = 1;


// Paths 
fs::path vocPath("../ORBvoc/ORBvoc.txt");
fs::path dbPath("../demo/Database/");
fs::path queryPath("../demo/Query/");
fs::path savePath("../demo/Result/");


int main()
{
  EASY_MAIN_THREAD;
  EASY_PROFILER_ENABLE;

  // DataStructures
  SampleCode::FileLUT queryResultTable;
  SampleCode::FileLUT dbTable;
  SampleCode::FeatureVector dbFeatures;
  SampleCode::FeatureVector queryFeatures;
  cv::Ptr<cv::ORB> ORB = cv::ORB::create();
  std::unique_ptr<OrbVocabulary> ptrVocabulary(new OrbVocabulary());

  if(!loadDBFeatures(dbFeatures, dbTable, ORB))
    std::cerr << "Error while loading DB features\n" << std::endl;

  if(!loadVocabulary(*ptrVocabulary))
    std::cerr << "Error while loading Vocabulary\n" << std::endl;

  EASY_BLOCK("Initialize Database", profiler::colors::LightBlue);
  std::unique_ptr<OrbDatabase> ptrDatabase(new OrbDatabase(*ptrVocabulary, false, 0));
  EASY_END_BLOCK;

  if(!createDatabase(*ptrDatabase, dbFeatures))
    std::cerr << "Error while Creating DB\n" << std::endl;

  if(!loadQueryFeatures(queryFeatures, ORB))
    std::cerr << "Error loading Query features\n" << std::endl;

  if(!queryDatabase(*ptrDatabase, queryFeatures, queryResultTable, dbTable))
    std::cerr << "Error while Querying DB\n" << std::endl;

  if(!saveResult(queryResultTable))
    std::cerr << "Error while saving Results\n" << std::endl; 

  profiler::dumpBlocksToFile("DBoW2.prof");
  return 0;
}


bool loadDBFeatures(SampleCode::FeatureVector& vfeatures, SampleCode::FileLUT& dbTable, const cv::Ptr<cv::ORB>& orb)
{
  EASY_FUNCTION("Load DB Features", profiler::colors::LightGreen);

  vfeatures.clear();
  vfeatures.reserve(NDBIMAGES);

  for(fs::directory_iterator it(dbPath); it != fs::end(it); it++)
  {
    const fs::directory_entry& entry = *it;
    const std::string fileName = entry.path();
    cv::Mat image = cv::imread(fileName, 0);
    if(image.empty()) 
      return false;

    cv::Mat mask;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    dbTable.push_back(fileName);
    vfeatures.emplace_back(SampleCode::Features());
    changeStructure(descriptors, vfeatures.back());
  }

  return true;
  
}


void changeStructure(const cv::Mat& plain, SampleCode::Features& out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
    out[i] = plain.row(i);
  
}


bool loadVocabulary(OrbVocabulary& voc)
{
  EASY_FUNCTION("Load Vocabulary", profiler::colors::Yellow);
  
  if(!voc.loadFromTextFile(vocPath))  
    return false;

  std::cout << "< Vocabulary information >" << std::endl
  << voc << std::endl << std::endl;

  return true;

}


bool createDatabase(OrbDatabase& db, const SampleCode::FeatureVector& vfeatures)
{
  EASY_FUNCTION("Create Database", profiler::colors::Magenta);

  for(int i = 0; i < NDBIMAGES; i++)
    db.add(vfeatures[i]);

  std::cout << "< Database information >\n" << db << std::endl << std::endl;

  return true;
}


bool loadQueryFeatures(SampleCode::FeatureVector& vfeatures, const cv::Ptr<cv::ORB>& orb)
{
  EASY_FUNCTION("Load Query Features", profiler::colors::DeepOrange);

  vfeatures.clear();
  vfeatures.reserve(NQUERYIMAGES);

  // Feature Extraction for Query Images 
  for(fs::directory_iterator it(queryPath); it != fs::end(it); it++)
  {
    const fs::directory_entry& entry = *it;
    const std::string fileName = entry.path();
    cv::Mat image = cv::imread(fileName, 0);
    if(image.empty()) 
      return false;

    cv::Mat mask;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    vfeatures.emplace_back(SampleCode::Features());
    changeStructure(descriptors, vfeatures.back());
  }

  return true;

}


bool queryDatabase(OrbDatabase& db, const SampleCode::FeatureVector& vfeatures, SampleCode::FileLUT& queryResultTable, const SampleCode::FileLUT& dbTable)
{
  EASY_FUNCTION("Query Database", profiler::colors::DarkTeal);

  QueryResults ret;
  int nCandidates = 1;
  if(nCandidates > NQUERYIMAGES)  
    return false; 

  for(int i = 0; i < NQUERYIMAGES; i++)
  {
    // Query & Save "nCandidates" best match in QueryResult
    db.query(vfeatures[i], ret, nCandidates);

    std::cout << "Searching for Image " << i << ". " << ret << std::endl << std::endl;

    // Save Image index of the Best Match for each Query Image
    const auto& bestMatch = dbTable[ret[0].Id];
    queryResultTable.push_back(bestMatch);
  }

  return true;

}


bool saveResult(const SampleCode::FileLUT& queryResultTable)
{
  // Search Image by index and Save 
  for(int i = 0; i < NQUERYIMAGES; i++)
  {
    cv::Mat image = cv::imread(queryResultTable[i], 0);
    if(image.empty()) 
      return false;

    std::string fileName = savePath;
    fileName = fileName + "Result" + std::to_string(i) + ".png";
    cv::imwrite(fileName, image);
  }

  return true;

}
