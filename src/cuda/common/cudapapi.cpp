#include <iostream>
#include "cudapapi.h"

using namespace std;

#define NUM_RECORDED_EVENTS 1
#define CUDA_PAPI_VERBOSE 0
#define NUM_EVENTS 9

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

volatile size_t currCudaPapiEvent;
CudaPapiCode cudaPapiState = Stopped;
int EventSet = PAPI_NULL;
int currEvent;

void _InitCudaPapi(void)
{
  int retval;
  currCudaPapiEvent = 6;

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

void _StartCounters(volatile int *cudaPapiTestVal)
{
  int retval;
  int eventCount = 0;
  int events[NUM_RECORDED_EVENTS];

  if (Error == cudaPapiState) {
    cerr << "Attempted to start event count, a pre-existing error state exists!" << endl;
    return;
  }

  if (Stopped == cudaPapiState) {
    cerr << "Attempted to start event count without initializing PAPI!" << endl;
    return;
  }

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

  retval = PAPI_start( EventSet );
  if( retval != PAPI_OK ) {
    CudaPapiWarning("PAPI_start failed", retval);
  }
  cudaPapiState = Started;
  *cudaPapiTestVal = cudaPapiState;
}

void _StopCounters(string testName, string attr, ResultDatabase &resultDB, volatile int *cudaPapiTestVal)
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

  if (Stopped == cudaPapiState) {
    cerr << "Attempted to stop event count without starting one!" << endl;
    return;
  }

  retval = PAPI_stop( EventSet, values );
  if( retval != PAPI_OK )
    CudaPapiError("PAPI_stop failed", retval );

  retval = PAPI_cleanup_eventset(EventSet);
  if( retval != PAPI_OK )
    CudaPapiError("PAPI_cleanup_eventset failed", retval );

  retval = PAPI_destroy_eventset(&EventSet);
  if (retval != PAPI_OK)
    CudaPapiError("PAPI_destroy_eventset failed", retval );

  allEvents[currCudaPapiEvent].eventCount = values[0];

  resultDB.AddResult(allEvents[currCudaPapiEvent].eventName,
                     testName + "_" + attr + "_" + to_string(currCudaPapiEvent),
                     "count",
                     values[0]);
  *cudaPapiTestVal = cudaPapiState;
}

void _CurrentCounterInfo(string& counterName, long long& count)
{
  counterName = string(allEvents[currCudaPapiEvent].eventName);
  count = allEvents[currCudaPapiEvent].eventCount;
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
