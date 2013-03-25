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

#include "ArrayManager.h"

ArrayManager::ArrayManager(): specific_dimensions(false), ndim_min(-1), ndim_max(-1),
                              array_obj(NULL), data(NULL), size(NULL)
{ }

ArrayManager::~ArrayManager() {
  if (array_obj)
    Py_DECREF(array_obj);
}

inline double& ArrayManager::operator[](npy_int j) { return data[j]; }
inline double& ArrayManager::operator()(npy_int x1) { return data[x1]; }
inline double& ArrayManager::operator()(npy_int x2, npy_int x1) { return data[x2 * size[1] + x1]; }
inline double& ArrayManager::operator()(npy_int x3, npy_int x2, npy_int x1) {
  return data[((x3 * size[2]) + x2) * size[1] + x1]; }
inline double& ArrayManager::operator()(npy_int x4, npy_int x3, npy_int x2, npy_int x1) {
  return data[(((x4 * size[3] + x3) * size[2]) + x2) * size[1] + x1]; }

ArrayManager& ArrayManager::operator=(PyObject* obj) {
  array_obj = (PyArrayObject*) PyArray_FROM_OTF(obj, NPY_INOUT_ARRAY, NPY_FLOAT64);
  if (array_obj == NULL) throw std::exception();
  
  if (specific_dimensions) {
    if (array_obj->nd < ndim_min) { 
      char err_msg[1024];
      snprintf(err_msg, 1024, "Incorrect number of dimensions (%d < %d)",
               array_obj->nd, ndim_min);
      throw std::runtime_error(err_msg);          
    }
    else if (array_obj->nd > ndim_max) {
      char err_msg[1024];
      snprintf(err_msg, 1024, "Incorrect number of dimensions (%d > %d)",
             array_obj->nd, ndim_max);
      throw std::runtime_error(err_msg);          
    }
  }
  
  size = array_obj->dimensions;
  data = (double*) array_obj->data;

  return *this;
}

void ArrayManager::set_dimensions(int input_ndim) {
  // This method must be called on a non-initialized object.
  if (data != NULL) throw std::exception();

  specific_dimensions = true;
  ndim_min = input_ndim;
  ndim_max = input_ndim;
}

void ArrayManager::set_dimensions(int input_ndim_min, int input_ndim_max) {
  // This method must be called on a non-initialized object.
  if (data != NULL) throw std::exception();
  if (input_ndim_min > input_ndim_max) throw std::exception();

  specific_dimensions = true;
  ndim_min = input_ndim_min;
  ndim_max = input_ndim_max;
}
