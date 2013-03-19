/* Copyright 2013 Nicolau Leal Werneck
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

#ifndef ARRAYMANAGER_H
#define ARRAYMANAGER_H
#include <Python.h>
#include "numpy/arrayobject.h"

class ArrayManager : public TupleStreamExtractable {
public:
  PyArrayObject *array_obj;

  // Pointer to the actual data array from array_obj.
  double* data;

  // Size of the array in each dimension. Simply points to the
  // "dimensions" from the array_obj. "size" is the name used in
  // Python.
  npy_int* size;

  ArrayManager();

  // The main reason for this class to be created. Running PyDECREF on
  // the array reference at the object destructor.
  ~ArrayManager();

  // Square brackets simply gives access to the "data" pointer from
  // the array.
  inline double& operator[](npy_int);

  // Overload the parenthesis to allow for indexing with multiple
  // dimensions. Remember square brackets can never receive multiple
  // arguments, so this is why we used the parenthesis. In practice
  // you can sometimes save some multiplications by handling the index
  // calculation yourself, and using [], but these methods here can be
  // very handy, and can often be fast enough.  And final observation:
  // there are no bound checks whatsoever. This class is only supposed
  // to help you access the data from numpy arrays to perform fast
  // calculations. You need to know what you are doing.
  inline double& operator()(npy_int);
  inline double& operator()(npy_int,npy_int);
  inline double& operator()(npy_int,npy_int,npy_int);
  inline double& operator()(npy_int,npy_int,npy_int,npy_int);

  ArrayManager& operator=(PyObject* obj);
};
#endif /* ARRAYMANAGER_H */
