/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <iomanip>


// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

#include <easy/profiler.h>

using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadDBFeatures(vector<vector<cv::Mat > > &features);
void loadVocabulary(OrbVocabulary* vobabulary);
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out);
void createDatabase(OrbDatabase* database, const vector<vector<cv::Mat > > &features);
void loadQueryFeatures(vector<vector<cv::Mat > > &features);
void queryDatabase(OrbDatabase* database, const vector<vector<cv::Mat > > &features, vector<int> &imgIdx);
void saveResult(vector<int> &imgIdx);

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// Number of Images to Build Database
const int NDBIMAGES = 4;
// Number of Images to Query with Database
const int NQUERYIMAGES = 1;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// Paths 
const string vocPath = "../ORBvoc/ORBvoc.txt";
const string dbPath = "../demo/Database/";
const string queryPath = "../demo/Query/";


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  cout << endl << "Press enter to save results..." << endl;
  getchar();
}

// ----------------------------------------------------------------------------

int main()
{
  // Initiate Profiler
  EASY_MAIN_THREAD;
  EASY_PROFILER_ENABLE;

  // DataStructures
  vector<int> imgIndex;
  vector<vector<cv::Mat > > dbFeatures;
  vector<vector<cv::Mat > > queryFeatures;
  OrbVocabulary* ptrVocabulary = new OrbVocabulary();

  loadDBFeatures(dbFeatures);

  // Load Vocabulary
  loadVocabulary(ptrVocabulary);
  
  EASY_BLOCK("Initialize Database", profiler::colors::LightBlue);
  OrbDatabase* ptrDatabase = new OrbDatabase(*ptrVocabulary, false, 0);
  EASY_END_BLOCK;

  createDatabase(ptrDatabase, dbFeatures);

  loadQueryFeatures(queryFeatures);

  queryDatabase(ptrDatabase, queryFeatures, imgIndex);

  wait();

  saveResult(imgIndex);

  profiler::dumpBlocksToFile("DBoW2.prof");
  return 0;
}

// ----------------------------------------------------------------------------

void loadDBFeatures(vector<vector<cv::Mat > > &features)
{
  EASY_FUNCTION("Load DB Features", profiler::colors::LightGreen);

  features.clear();
  features.reserve(NDBIMAGES);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << " < Feature Extraction for DB Images >" << endl;
  for(int i = 0; i < NDBIMAGES; ++i)
  {
    //if(i % 500 == 0) cout << "Extracting Features from " << i << "th Image ... " << endl; 

    stringstream ss;
    ss << dbPath << setfill('0') << setw(6) << i << ".png";

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

void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
  out.resize(plain.rows);

  for(int i = 0; i < plain.rows; ++i)
  {
    out[i] = plain.row(i);
  }
}

// ----------------------------------------------------------------------------

void loadVocabulary(OrbVocabulary* voc)
{
  EASY_FUNCTION("Load Vocabulary", profiler::colors::Yellow);

  cout << endl << "< Loading ORB Vocabulary >" << endl;

  bool bVocLoad = voc->loadFromTextFile(vocPath);
  if(!bVocLoad)
  {
      cerr << "Wrong path to vocabulary. " << endl;
      cerr << "Falied to open at: " << vocPath << endl;
      exit(-1);
  }

  cout << endl << "< Vocabulary information >" << endl
  << *voc << endl << endl;

  return;
}

// ----------------------------------------------------------------------------

void createDatabase(OrbDatabase* db, const vector<vector<cv::Mat > > &features)
{
  EASY_FUNCTION("Create Database", profiler::colors::Magenta);

  cout << "< Creating a Database >" << endl;

  // add images to the database
  for(int i = 0; i < NDBIMAGES; i++)
  {
    int idx = db->add(features[i]);
    //if(idx % 500 == 0) cout << "Adding " << idx << "th image to Database ... " << endl;
  }

  cout << endl << "< Database information >" << endl << *db << endl << endl;

  return;
}

// ----------------------------------------------------------------------------

void loadQueryFeatures(vector<vector<cv::Mat > > &features)
{
  EASY_FUNCTION("Load Query Features", profiler::colors::DeepOrange);

  features.clear();
  features.reserve(NQUERYIMAGES);

  cv::Ptr<cv::ORB> orb = cv::ORB::create();

  cout << "< Feature Extraction for Query Images > " << endl << endl;
  for(int i = 0; i < NQUERYIMAGES; ++i)
  {
    stringstream ss;
    
    ss << queryPath << setfill('0') << setw(6) << i << ".png";

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

void queryDatabase(OrbDatabase* db, const vector<vector<cv::Mat > > &features, vector<int> &imgIdx)
{
  EASY_FUNCTION("Query Database", profiler::colors::DarkTeal);

  cout << "< Querying the database > " << endl;

  QueryResults ret;
  for(int i = 0; i < NQUERYIMAGES; i++)
  {
    // Query & Save 4 best match in QueryResult
    db->query(features[i], ret, 4);

    cout << "Searching for Image " << i << ". " << ret << endl << endl;

    // Save Image index of the Best Match for each Query Image
    imgIdx.push_back(ret[0].Id);
  }

  cout << endl;

  return;
}

// ----------------------------------------------------------------------------

void saveResult(vector<int> &imgIdx)
{
  // Search Image by index and Save 
  for(int i = 0; i < NQUERYIMAGES; i++)
  {
    stringstream ssR;
    ssR << dbPath << setfill('0') << setw(6) << imgIdx[i] << ".png";
    cv::Mat image = cv::imread(ssR.str(), 0);

    stringstream ssW;
    ssW << queryPath << "Result" << i << ".png";
    cv::imwrite(ssW.str(), image);
  }

  return;
}
