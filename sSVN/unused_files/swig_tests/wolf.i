%module wolf

%{
	#define SWIG_FILE_WITH_INIT
	#include "wolf.h"
%}

%include "numpy.i"

%init %{
	import_array();
%}

%apply (double* IN_ARRAY1, int DIM1) {(double* x, int n)}

%include "wolf.h"