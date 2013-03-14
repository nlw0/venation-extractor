#include <Python.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include <exception>

class ArrayManager {
public:
  PyObject *py_object;
  PyArrayObject *array_obj;
  
  ArrayManager() {
  }
  
  void get_stuff(double **alpha, npy_int *j, npy_int *k) {
    array_obj = (PyArrayObject *)
      PyArray_ContiguousFromAny(py_object, PyArray_DOUBLE, 2, 2);
    if (array_obj == NULL) {
      throw std::exception();
    }    

    *alpha = (double*) array_obj->data;
    *j = array_obj->dimensions[0];
    *k = array_obj->dimensions[1];
  }

  void get_stuff(double **ptr) {
    array_obj = (PyArrayObject *)
      PyArray_ContiguousFromAny(py_object, PyArray_DOUBLE, 2, 2);
    if (array_obj == NULL) {
      throw std::exception();
    }    

    *ptr = (double*) array_obj->data;
  }

  ~ArrayManager(){
    Py_DECREF(array_obj);
  }

};

static PyObject * smooth(PyObject *self, PyObject *args) {
  ArrayManager alphaM;
  ArrayManager betaM;

  if (!PyArg_ParseTuple(args, "OO", &betaM.py_object, &alphaM.py_object))
    return NULL;

  double* alpha;
  npy_int Jalpha, Kalpha;
  alphaM.get_stuff(&alpha, &Jalpha, &Kalpha);

  double* beta;
  betaM.get_stuff(&beta);
  

  /* Answer each item according to a Bernoulli trial, with the
     appropriate probabilities. */
  npy_int j, k;
  for(j = 1; j < Jalpha-1; j++) {
    for(k = 1; k < Kalpha-1; k++) {
      beta[j*Kalpha+k] = (alpha[j*Kalpha+k] + alpha[j*Kalpha+k-1] + alpha[j*Kalpha+k+1] +
                          alpha[(j-1)*Kalpha+k] + alpha[(j+1)*Kalpha+k])/5.0;
    }    
  }

  return PyFloat_FromDouble(0.0);
}

static PyMethodDef IrtEmLowMethods[] =
  {
    {"smooth", smooth, METH_VARARGS,
     "Silly smooth filter.\n"},
    {NULL, NULL, 0, NULL}
  };

PyMODINIT_FUNC

initlowlevel(void) {
  (void) Py_InitModule("lowlevel", IrtEmLowMethods);
  import_array();
}
