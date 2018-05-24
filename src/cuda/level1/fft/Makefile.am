if BUILD_MPI
    MAYBE_MPI = epmpi
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
serialcuda_PROGRAMS = FFT

# How to build those programs?
FFT_SOURCES = FFT.cpp \
                main.cpp  $(MAYBE_CUDAPAPI)
FFT_LDADD = fftlib.o $(CUDA_LIBS) $(LIBS)
