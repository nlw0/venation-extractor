#include <Python.h>
#include <math.h>
#include "numpy/arrayobject.h"
#include <exception>
#include <string>

class ArrayManager {
public:
  PyObject *py_object;
  PyArrayObject *array_obj;

  npy_int* size;
  double* data;
  

  // ArrayManager() {
  // }

  void _initialize() {
    array_obj = (PyArrayObject *)
      PyArray_ContiguousFromAny(py_object, PyArray_DOUBLE, 2, 2);
    if (array_obj == NULL) {
      throw std::exception();
    }    

    size = array_obj->dimensions;
    data = (double*) array_obj->data;
  }

  ~ArrayManager(){
    Py_DECREF(array_obj);
  }

  inline double& operator()(npy_int j, npy_int k) {
    return data[j * size[1] + k];
  }

};

npy_int args_manager(PyObject *args, ArrayManager *a, ArrayManager *b) {
  if (!PyArg_ParseTuple(args, "OO", &a->py_object, &b->py_object))
    return 0;
  else {
    a->_initialize();
    b->_initialize();
    return 1;
  }
}






class ArgsParser {
  int count;
  int Nargs;
  PyObject *args;
  bool error;
public:
  
  friend ArgsParser& operator>>(ArgsParser &, long int &);

  class ArgsCountException: public std::exception {};
  class TypeErrorException: public std::exception {};
  class ErrorStateException: public std::exception {};

  ArgsParser(PyObject *input_args) {
    error = false;
    args = input_args;
    Nargs = PyTuple_Size(args);
    count = 0;
  }

  bool check_error() {
    return error;
  }

  void set_nargs_error() {
    PyErr_SetString(PyExc_TypeError, "Too few arguments provided.");
    error = true;
  }

  void set_type_error() {
    char err_msg[1024];
    snprintf(err_msg, 1024, "Incorrect type for parameter number %d.", count + 1);
    PyErr_SetString(PyExc_TypeError, err_msg);
    error = true;
  }

};

ArgsParser& operator>>(ArgsParser &input, long int &x) {
  try {
    if (input.count >= input.Nargs) throw ArgsParser::ArgsCountException();

    if (input.error) throw ArgsParser::ErrorStateException();

    PyObject *po = PyTuple_GetItem(input.args, input.count);

    if (!PyInt_CheckExact(po)) throw ArgsParser::TypeErrorException();

    x = PyInt_AsLong(po);    

    if ((x == -1) && PyErr_Occurred()) throw ArgsParser::TypeErrorException();
  }
  catch (ArgsParser::ArgsCountException) { input.set_nargs_error(); }
  catch (ArgsParser::TypeErrorException) { input.set_type_error(); }
  catch (ArgsParser::ErrorStateException) { }
  
  input.count++;
  return input;
}




static PyObject * test_function1(PyObject *self, PyObject *args) {
  ArgsParser qq(args);
  long int aa;
  long int bb;

  qq >> aa >> bb;

  if (qq.check_error())
    return NULL;

  return PyInt_FromLong(aa + bb);
}












static PyObject * smooth(PyObject *self, PyObject *args) {
  ArrayManager alpha;
  ArrayManager beta;

  args_manager(args, &beta, &alpha);

  npy_int j, k;
  for(j = 1; j < alpha.size[0] - 1; j++) {
    for(k = 1; k < alpha.size[1] - 1; k++) {
      beta(j, k) = (alpha(j, k) + alpha(j, k - 1) + alpha(j, k + 1) +
                     alpha(j - 1, k) + alpha(j + 1, k))/5.0;
    }    
  }

  return PyInt_FromLong(1);
}



static PyMethodDef IrtEmLowMethods[] =
  {
    {"smooth", smooth, METH_VARARGS,
     "Silly smooth filter.\n"},
    {"tf1", test_function1, METH_VARARGS,
     "test function 1.\n"},
    {NULL, NULL, 0, NULL}
  };

PyMODINIT_FUNC

initlowlevel(void) {
  (void) Py_InitModule("lowlevel", IrtEmLowMethods);
  import_array();
}
