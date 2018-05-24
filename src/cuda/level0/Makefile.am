if BUILD_MPI
    MAYBE_MPI = epmpi
endif

if BUILD_CUDAPAPI
  MAYBE_CUDAPAPI = cudapapi.cpp
  MAYBE_CUDAPAPI_FLAGS = -DCUDAPAPI
endif

SUBDIRS	= $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../common

AM_LDFLAGS = $(CUDA_LDFLAGS) $(MAYBE_CUDAPAPI_FLAGS)
AM_CPPFLAGS = $(CUDA_INC)

# What is the destination for programs built from this directory?
serialcudadir = $(bindir)/Serial/CUDA

# What programs should be installed in the bin/Serial/CUDA destination?
serialcuda_PROGRAMS = BusSpeedDownload \
						BusSpeedReadback \
						DeviceMemory \
						MaxFlops

BusSpeedDownload_SOURCES	= main.cpp $(MAYBE_CUDAPAPI)
BusSpeedDownload_LDADD		= BusSpeedDownload.o $(CUDA_LIBS) $(LIBS)

BusSpeedReadback_SOURCES	= main.cpp $(MAYBE_CUDAPAPI)
BusSpeedReadback_LDADD		= BusSpeedReadback.o $(CUDA_LIBS) $(LIBS)

DeviceMemory_SOURCES	= main.cpp $(MAYBE_CUDAPAPI)
DeviceMemory_LDADD		= DeviceMemory.o $(CUDA_LIBS) $(LIBS)

MaxFlops_SOURCES	= main.cpp $(MAYBE_CUDAPAPI)
MaxFlops_LDADD		= MaxFlops.o $(CUDA_LIBS) $(LIBS)
