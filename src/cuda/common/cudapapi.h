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
void _StartCounters(void);
void _StopCounters(string testName, string attr, ResultDatabase &resultDB);
void _CurrentCounterInfo(string& counterName, long long& count);
void _ResetCounts(void);
int  _GetNumEvents(void);

// Only allow the calls if cudapapi is enabled
#ifdef CUDAPAPI
  #define InitCudaPapi _InitCudaPapi
  #define ShutdownCudaPapi_ ShutdownCudaPapi
  #define StartCounters _StartCounters
  #define StopCounters _StopCounters
  #define CurrentCounterInfo _CurrentCounterInfo
  #define ResetCounts _ResetCounts
  #define GetNumEvents _GetNumEvents
#else
  #define InitCudaPapi()
  #define ShutdownCudaPapi()
  #define StartCounters()
  #define StopCounters(x,y,z)
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
extern volatile size_t currCudaPapiEvent;

#ifdef CUDAPAPI
  #define StartPapiCounts() for(currCudaPapiEvent=0; currCudaPapiEvent < GetNumEvents(); currCudaPapiEvent++){\
      StartCounters();
  #define StopPapiCounts(testName, testAttrs, resultDB) StopCounters(testName, testAttrs, resultDB);}
#else
  #define StartPapiCounts()
  #define StopPapiCounts(testName, testAttrs, resultDB)
#endif


// event struct
typedef struct CudaEvent_st {
  long long eventCount;
  const char eventName[100]; // papi use C strings so this is a char
                  // also compiler crashes if you leave it as variable size
} CudaEvent;

enum CudaPapiCode {Error = 0, Init, Started, Stopped};

#endif // CUDAPAPI_H
