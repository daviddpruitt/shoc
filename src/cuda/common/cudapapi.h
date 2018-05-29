#ifndef CUDAPAPI_H
#define CUDAPAPI_H

#include <string>
#include "ResultDatabase.h"

extern "C"
{
  #include "papi.h"
}

using namespace std;

void _InitCudaPapi(void);
void _ShutdownCudaPapi(void);
void _SetupCounters(void);
void _TeardownCounters(string testName, string attr, ResultDatabase &resultDB);
void _StartCounters(void);
void _StopCounters(void);
void _AccumulateCounters(void);
void _CurrentCounterInfo(string& counterName, long long& count);
void _ResetCounts(void);
int  _GetNumEvents(void);

// Only allow the calls if cudapapi is enabled
#ifdef CUDAPAPI
  #define InitCudaPapi _InitCudaPapi
  #define ShutdownCudaPapi_ ShutdownCudaPapi
  #define SetupCounters _SetupCounters
  #define TeardownCounters _TeardownCounters
  #define StartCounters _StartCounters
  #define StopCounters _StopCounters
  //#define AccumulateCounters _AccumulateCounters
  #define CurrentCounterInfo _CurrentCounterInfo
  #define ResetCounts _ResetCounts
  #define GetNumEvents _GetNumEvents
#else
  #define InitCudaPapi()
  #define ShutdownCudaPapi()
  #define SetupCounters()
  #define TeardownCounters(x,y,z)
  #define StartCounters()
  #define StopCounters()
  //#define AccumulateCounters()
  #define CurrentCounterInfo()
  #define ResetCounts()
  #define GetNumEvents()
#endif


// print error message to std error, set error flag and return
#define CudaPapiError(msg, retval)                                  \
{                                                                   \
    cerr << msg << ": " <<  PAPI_strerror(retval) << endl;          \
    cudaPapiState = Error;                                          \
    return;                                                         \
}

#define CudaPapiWarning(msg, retval)                                \
{                                                                   \
    cerr << msg << ": " <<  PAPI_strerror(retval) << endl;          \
    cudaPapiState = Init;                                           \
    return;                                                         \
}

// These are macros instead of functions, that way the caller doesn't have to
// worry about cycling through all the events, we'll take care of it
extern size_t currCudaPapiEvent;

#ifdef CUDAPAPI
// #define StartPapiCountRegion(cudaPapiTestVal) _StartCounters(cudaPapiTestVal)
// #define StopPapiCountRegion(testName, testAttrs, resultDB, cudaPapiTestVal) _StopCounters(testName, testAttrs, resultDB, cudaPapiTestVal)
 #define StartPapiCountRegion() for(currCudaPapiEvent=0; currCudaPapiEvent < GetNumEvents(); currCudaPapiEvent++){_SetupCounters();
 #define EndPapiCountRegion(testName, testAttrs, resultDB) _TeardownCounters(testName, testAttrs, resultDB);}
#else
  #define StartPapiCountRegion()
  #define EndPapiCountRegion(testName, testAttrs, resultDB)
#endif


// event struct
typedef struct CudaEvent_st {
  long long eventCount;
  const char eventName[100]; // papi use C strings so this is a char
                  // also compiler crashes if you leave it as variable size
} CudaEvent;

enum CudaPapiCode {Error = 0, Init, Created, Started, Stopped};

#endif // CUDAPAPI_H
