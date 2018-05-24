include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

if BUILD_CUDAPAPI
  MAYBE_CUDAPAPI = cudapapi.cpp
  MAYBE_CUDAPAPI_FLAGS = -DCUDAPAPI
endif

# How to find source files
VPATH = $(srcdir)/..:$(srcdir)/../../../common:$(srcdir)/../../../../common

# Which compiler to use to build and link
CXX = ${MPICXX}
CXXLD = ${MPICXX}

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC) $(MPI_CPPFLAGS) -I$(top_srcdir)/src/mpi/common $(MAYBE_CUDAPAPI_FLAGS)

# What is the destination for programs built from this directory?
tpcudadir = $(bindir)/TP/CUDA

# What programs should be installed to that destination?
tpcuda_PROGRAMS = QTC

# How to build those programs?
QTC_SOURCES = comm.cpp \
				libdata.cpp \
                main.cpp $(MAYBE_CUDAPAPI)
QTC_LDADD = QTC.o $(CUDA_LIBS) $(LIBS)
