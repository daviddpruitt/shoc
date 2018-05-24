include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

if BUILD_CUDAPAPI
  MAYBE_CUDAPAPI = cudapapi.cpp
  MAYBE_CUDAPAPI_FLAGS = -DCUDAPAPI
endif

VPATH = $(srcdir)/..:$(srcdir)/../../../common:$(srcdir)/../../../../common

# Which compiler to use to build and link
CXX = ${MPICXX}
CXXLD = ${MPICXX}

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC) $(MPI_CPPFLAGS) $(MAYBE_CUDAPAPI_FLAGS) -I$(top_srcdir)/src/mpi/common

# What is the destination for programs built from this directory?
tpcudadir = $(bindir)/TP/CUDA

# What programs should be installed to that destination?
tpcuda_PROGRAMS = Stencil2D

# How to build those programs?
Stencil2D_SOURCES = CUDAStencil.cpp \
				CommonCUDAStencilFactory.cpp \
				Stencil2Dmain.cpp \
				CUDAStencilFactory.cpp \
                main.cpp $(MAYBE_CUDAPAPI)
Stencil2D_LDADD = CUDAStencilKernel.o $(CUDA_LIBS) $(LIBS)
