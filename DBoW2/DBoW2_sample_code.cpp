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

using DescriptorVector = std::vector<std::vector<cv::Mat>>;
using Descriptors = std::vector<cv::Mat>;
using FileLUT = std::vector<std::string>;

bool loadDBFeatures(DescriptorVector& vfeatures, FileLUT& dbTable, const cv::Ptr<cv::ORB>& orb);
bool loadVocabulary(const std::unique_ptr<OrbVocabulary>& voc);
void changeStructure(const cv::Mat& plain, Descriptors& out);
bool createDatabase(const std::unique_ptr<OrbDatabase>& db, const DescriptorVector& vfeatures);
bool loadQueryFeatures(DescriptorVector& vfeatures, const cv::Ptr<cv::ORB>& orb);
bool queryDatabase(const std::unique_ptr<OrbDatabase>& db, const DescriptorVector& vfeatures, FileLUT& queryResultTable, const FileLUT& dbTable);
bool saveResult(const FileLUT& queryResultTable);


// Number of Images to Build Database
constexpr int NDBIMAGES = 4;
// Number of Images to Query with Database
constexpr int NQUERYIMAGES = 1;


// Paths 
fs::path curPath = fs::current_path();
fs::path vocPath = curPath.parent_path() / "ORBvoc" / "ORBvoc.txt";
fs::path dbPath = curPath.parent_path() / "demo" / "Database";
fs::path queryPath = curPath.parent_path() / "demo" / "Query";
fs::path savePath = curPath.parent_path() / "demo" / "Result" / "";



int main()
{
  EASY_MAIN_THREAD;
  EASY_PROFILER_ENABLE;

  // DataStructures
  FileLUT queryResultTable;
  FileLUT dbTable;
  DescriptorVector dbFeatures;
  DescriptorVector queryFeatures;
  cv::Ptr<cv::ORB> ORB = cv::ORB::create();
  auto ptrVocabulary = std::make_unique<OrbVocabulary>();

  if(!loadDBFeatures(dbFeatures, dbTable, ORB))
    std::cerr << "Error while loading DB features\n" << std::endl;

  if(!loadVocabulary(ptrVocabulary))
    std::cerr << "Error while loading Vocabulary\n" << std::endl;

  EASY_BLOCK("Initialize Database", profiler::colors::LightBlue);
  auto& vocabulary = *ptrVocabulary;
  auto ptrDatabase = std::make_unique<OrbDatabase>(vocabulary);
  EASY_END_BLOCK;

  if(!createDatabase(ptrDatabase, dbFeatures))
    std::cerr << "Error while Creating DB\n" << std::endl;

  if(!loadQueryFeatures(queryFeatures, ORB))
    std::cerr << "Error loading Query features\n" << std::endl;

  if(!queryDatabase(ptrDatabase, queryFeatures, queryResultTable, dbTable))
    std::cerr << "Error while Querying DB\n" << std::endl;

  if(!saveResult(queryResultTable))
    std::cerr << "Error while saving Results\n" << std::endl; 

  profiler::dumpBlocksToFile("DBoW2.prof");
  return 0;
}


bool loadDBFeatures(DescriptorVector& vfeatures, FileLUT& dbTable, const cv::Ptr<cv::ORB>& orb)
{
  EASY_FUNCTION("Load DB Features", profiler::colors::LightGreen);

  vfeatures.clear();
  vfeatures.reserve(NDBIMAGES);

  for(const auto& entry : fs::directory_iterator(dbPath))
  {
    const std::string fileName = entry.path();
    const cv::Mat image = cv::imread(fileName, 0);
    if(image.empty()) 
      return false;

    cv::Mat mask;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    dbTable.push_back(fileName);
    vfeatures.emplace_back(Descriptors());
    changeStructure(descriptors, vfeatures.back());
  }

  return true;
}


void changeStructure(const cv::Mat& plain, Descriptors& out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
    out[i] = plain.row(i);
}


bool loadVocabulary(const std::unique_ptr<OrbVocabulary>& voc)
{
  EASY_FUNCTION("Load Vocabulary", profiler::colors::Yellow);
  if(!voc->loadFromTextFile(vocPath))  
    return false;

  std::cout << "< Vocabulary information >" << std::endl
  << *voc << std::endl << std::endl;

  return true;
}


bool createDatabase(const std::unique_ptr<OrbDatabase>& db, const DescriptorVector& vfeatures)
{
  EASY_FUNCTION("Create Database", profiler::colors::Magenta);

  for(int i = 0; i < NDBIMAGES; i++)
    db->add(vfeatures[i]);

  std::cout << "< Database information >\n" << *db << std::endl << std::endl;

  return true;
}


bool loadQueryFeatures(DescriptorVector& vfeatures, const cv::Ptr<cv::ORB>& orb)
{
  EASY_FUNCTION("Load Query Features", profiler::colors::DeepOrange);

  vfeatures.clear();
  vfeatures.reserve(NQUERYIMAGES);

  // Feature Extraction for Query Images 
  for(const auto& entry : fs::directory_iterator(queryPath))
  {
    const std::string fileName = entry.path();
    const cv::Mat image = cv::imread(fileName, 0);
    if(image.empty()) 
      return false;

    cv::Mat mask;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(image, mask, keypoints, descriptors);

    vfeatures.emplace_back(Descriptors());
    changeStructure(descriptors, vfeatures.back());
  }

  return true;
}


bool queryDatabase(const std::unique_ptr<OrbDatabase>& db, const DescriptorVector& vfeatures, FileLUT& queryResultTable, const FileLUT& dbTable)
{
  EASY_FUNCTION("Query Database", profiler::colors::DarkTeal);

  QueryResults ret;
  int nCandidates = 1;
  if(nCandidates > NQUERYIMAGES)  
    return false; 

  for(int i = 0; i < NQUERYIMAGES; i++)
  {
    // Query & Save "nCandidates" best match in QueryResult
    db->query(vfeatures[i], ret, nCandidates);

    std::cout << "Searching for Image " << i << ". " << ret << std::endl << std::endl;

    // Save Image index of the Best Match for each Query Image
    const auto& bestMatch = dbTable[ret[0].Id];
    queryResultTable.push_back(bestMatch);
  }

  return true;
}


bool saveResult(const FileLUT& queryResultTable)
{
  // Search Image by index and Save 
  for(int i = 0; i < NQUERYIMAGES; i++)
  {
    const cv::Mat image = cv::imread(queryResultTable[i], 0);
    if(image.empty()) 
      return false;

    std::string fileName = savePath;
    fileName = fileName + "Result" + std::to_string(i) + ".png";
    cv::imwrite(fileName, image);
  }

  return true;
}
