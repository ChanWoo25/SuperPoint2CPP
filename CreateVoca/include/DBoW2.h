
#ifndef __D_T_DBOW2__
#define __D_T_DBOW2__

/// Includes all the data structures to manage vocabularies and image databases
namespace DBoW2
{
}

#include "TemplatedVocabulary.h"
#include "TemplatedDatabase.h"
#include "BowVector.h"
#include "FeatureVector.h"
#include "QueryResults.h"
#include "FSuperpoint.h"

/// ORB Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FSUPERPOINT::TDescriptor, DBoW2::FSUPERPOINT> 
  SuperpointVocabulary;

/// FORB Database
typedef DBoW2::TemplatedDatabase<DBoW2::FSUPERPOINT::TDescriptor, DBoW2::FSUPERPOINT> 
  SuperpointDatabase;

#endif

