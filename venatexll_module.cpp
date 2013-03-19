#include <Python.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include <exception>
#include <string>

#include "PyTupleStream.h"
#include "ArrayManager.h"

#include "PyTupleStream.cpp"
#include "ArrayManager.cpp"


static PyObject * smooth(PyObject *self, PyObject *args)
{
  TupleStream qq(args);
  
  ArrayManager alpha;
  ArrayManager beta;
  
  qq >> alpha >> beta;
  
  if (qq.fail()) return NULL;
  
  npy_int j, k;
  for(j = 1; j < alpha.size[0] - 1; j++) {
    for(k = 1; k < alpha.size[1] - 1; k++) {
      beta(j, k) = (alpha(j, k) + alpha(j, k - 1) + alpha(j, k + 1) +
                    alpha(j - 1, k) + alpha(j + 1, k))/5.0;
    }    
  }
  
  return PyInt_FromLong(1);
}


static PyMethodDef VenatExLowLevelMethods[] =
  {
    {"smooth", smooth, METH_VARARGS,
     "Silly smooth filter.\n"},
    {NULL, NULL, 0, NULL}
  };

PyMODINIT_FUNC

initvenatexll(void) {
  (void) Py_InitModule("venatexll", VenatExLowLevelMethods);
  import_array();
}
