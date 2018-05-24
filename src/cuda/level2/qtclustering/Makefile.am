if BUILD_MPI
    MAYBE_MPI = tpmpi
endif

if BUILD_CUDAPAPI
  MAYBE_CUDAPAPI = cudapapi.cpp
  MAYBE_CUDAPAPI_FLAGS = -DCUDAPAPI
endif

SUBDIRS = $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../common:$(srcdir)/../../../common

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC) $(MAYBE_CUDAPAPI_FLAGS)

# What is the destination for programs built from this directory?
serialcudadir = $(bindir)/Serial/CUDA

# What programs should be installed to that destination?
serialcuda_PROGRAMS = QTC

# How to build those programs?
QTC_SOURCES = comm.cpp \
				libdata.cpp \
                main.cpp $(MAYBE_CUDAPAPI)
QTC_LDADD = QTC.o $(CUDA_LIBS) $(LIBS)
