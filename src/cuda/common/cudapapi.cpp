#include <iostream>
#include "cudapapi.h"

using namespace std;

#define NUM_RECORDED_EVENTS 1
#define CUDA_PAPI_VERBOSE 0
#define NUM_EVENTS 9

#define checkNotInState(state, message)  if(state == cudaPapiState){cerr << message; return;}

CudaEvent allEvents[] = {
  {0, "cuda:::metric:local_store_throughput:device=0"  },
  {0, "cuda:::metric:local_load_throughput:device=0"   },

  {0, "cuda:::metric:shared_store_throughput:device=0" },
  {0, "cuda:::metric:shared_load_throughput:device=0"  },

  {0, "cuda:::metric:gst_requested_throughput:device=0"},
  {0, "cuda:::metric:gld_requested_throughput:device=0"},

  {0, "cuda:::event:inst_executed:device=0"    },
  {0, "cuda:::event:elapsed_cycles_sm:device=0"    },
  {0, "cuda:::metric:warp_execution_efficiency:device=0"   },

  {0, "cuda:::event:gld_inst_8bit:device=0"   },
  {0, "cuda:::event:gld_inst_16bit:device=0"   },
  {0, "cuda:::event:gld_inst_32bit:device=0"   },
  {0, "cuda:::event:gld_inst_64bit:device=0"   },
  {0, "cuda:::event:gld_inst_128bit:device=0"   },

  {0, "cuda:::event:gst_inst_8bit:device=0"   },
  {0, "cuda:::event:gst_inst_16bit:device=0"   },
  {0, "cuda:::event:gst_inst_32bit:device=0"   },
  {0, "cuda:::event:gst_inst_64bit:device=0"   },
  {0, "cuda:::event:gst_inst_128bit:device=0"   },

};

size_t currCudaPapiEvent;
CudaPapiCode cudaPapiState = Stopped;
int EventSet = PAPI_NULL;
int currEvent;

void _InitCudaPapi(void)
{
  int retval;
  //currCudaPapiEvent = 0;

  /* PAPI Initialization */
  retval = PAPI_library_init( PAPI_VER_CURRENT );
  if( retval != PAPI_VER_CURRENT ) {
    CudaPapiError("PAPI_library_init failed", retval);
  }

  // print papi version
  if (CUDA_PAPI_VERBOSE) {
    cout << "PAPI_VERSION " <<
      PAPI_VERSION_MAJOR( PAPI_VERSION )    << "." <<
      PAPI_VERSION_MINOR( PAPI_VERSION )    << "." <<
      PAPI_VERSION_REVISION( PAPI_VERSION ) << endl;
  }

  _ResetCounts();
  cudaPapiState = Init;
}

void _ShutdownCudaPapi(void)
{
  PAPI_shutdown();
}

void _SetupCounters(void)
{
  int retval;
  int eventCount = 0;
  int events[NUM_RECORDED_EVENTS];

  checkNotInState(Error, "Attempted to start event count, a pre-existing error state exists!");
  checkNotInState(Created, "Attempted to create event count while a previous event count exists!" );
  checkNotInState(Started, "Attempted to create event count while a previous event count is running!");
  checkNotInState(Stopped, "Attempted to setup event count without initializing PAPI!");

  // if (Error == cudaPapiState) {
  //   cerr << "Attempted to start event count, a pre-existing error state exists!" << endl;
  //   return;
  // }
  //
  // if (Stopped == cudaPapiState) {
  //   cerr << "Attempted to start event count without initializing PAPI!" << endl;
  //   return;
  // }
  //
  // if (Started == cudaPapiState) {
  //   cerr << "Attempted to create event count while a previous event count is running!" << endl;
  //   return;
  // }
  //
  // if (Created == cudaPapiState) {
  //   cerr << "Attempted to create event count while a previous event count exists!" << endl;
  //   return;
  // }

  /* convert PAPI native events to PAPI code */
  retval = PAPI_event_name_to_code(
                  (char *)allEvents[currCudaPapiEvent].eventName,
                  events );
  if( retval != PAPI_OK ) {
    CudaPapiWarning("PAPI_event_name_to_code failed, skipping", retval);
  }

  eventCount++;
  if (CUDA_PAPI_VERBOSE)
        cout << "Name " << allEvents[currCudaPapiEvent].eventName <<
                " --- " << "Code: " << events[0] << endl;

  retval = PAPI_create_eventset( &EventSet );
  if( retval != PAPI_OK ) {
    CudaPapiWarning("Cannot create eventset", retval);
  }

  // If multiple GPUs/contexts were being used,
  // you need to switch to each device before adding its events
  // e.g. cudaSetDevice( 0 );
  retval = PAPI_add_events( EventSet, events, eventCount );
  if( retval != PAPI_OK ) {
    CudaPapiWarning("PAPI_add_events failed", retval);
  }

  cudaPapiState = Created;
}

void _TeardownCounters(string testName, string attr, ResultDatabase &resultDB)
{
  long long values[1];
  int retval;

  if (Error == cudaPapiState) {
    cerr << "Attempted to stop event count when a previous error state exists!" << endl;
    return;
  }

  if (Init  == cudaPapiState) {
    cerr << "Attempted to stop event count when a previous count has not been started!" << endl;
    return;
  }

  if (Started == cudaPapiState) {
    _StopCounters();
  }

  retval = PAPI_cleanup_eventset(EventSet);
  if( retval != PAPI_OK )
    CudaPapiError("PAPI_cleanup_eventset failed", retval );

  retval = PAPI_destroy_eventset(&EventSet);
  if (retval != PAPI_OK)
    CudaPapiError("PAPI_destroy_eventset failed", retval );

  resultDB.AddResult(allEvents[currCudaPapiEvent].eventName,
                     testName + "_" + attr + "_" + to_string(currCudaPapiEvent),
                     "count",
                     allEvents[currCudaPapiEvent].eventCount);

  cudaPapiState = Init;
}

void _StartCounters(void)
{
  int retval;

  if (Error == cudaPapiState) {
    cerr << "Attempted to start count while an error condition exists (see previous messages)!" << endl;
    return;
  }

  if (Init == cudaPapiState) {
    cerr << "Attempted to start count without creating one!" << endl;
    return;
  }

  if (Started == cudaPapiState) {
    cerr << "Attempted to start count while a previous event count is running!" << endl;
    return;
  }

  retval = PAPI_start( EventSet );
  if( retval != PAPI_OK ) {
    CudaPapiWarning("PAPI_start failed", retval);
  }

  cudaPapiState = Started;
  return;
}
void _StopCounters(void)
{
  long long values[1];
  int retval;

  if (Error == cudaPapiState) {
    cerr << "Attempted to stop count while an error condition exists (see previous messages)!" << endl;
    return;
  }

  if (Init == cudaPapiState) {
    cerr << "Attempted to stop count without creating one!" << endl;
    return;
  }

  if (Stopped == cudaPapiState) {
    cerr << "Attempted to start count without starting one!" << endl;
    return;
  }

  retval = PAPI_stop( EventSet, values );
  if( retval != PAPI_OK )
    CudaPapiError("PAPI_stop failed", retval );

  allEvents[currCudaPapiEvent].eventCount += values[0];

  cudaPapiState = Stopped;
  return;
}

void _CurrentCounterInfo(string& counterName, long long& count)
{
  counterName = string(allEvents[currCudaPapiEvent].eventName);
  count = allEvents[currCudaPapiEvent].eventCount;

  cudaPapiState = Stopped;
}

void _ResetCounts(void)
{
  int idx;
  for (idx = 0; idx < NUM_EVENTS; idx++) {
    allEvents[idx].eventCount = 0;
  }
}

int _GetNumEvents(void)
{
  return NUM_EVENTS;
}
