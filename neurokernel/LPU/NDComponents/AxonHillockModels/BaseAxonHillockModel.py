#!/usr/bin/env python

from abc import ABCMeta, abstractmethod, abstractproperty
from future.utils import with_metaclass
from neurokernel.LPU.NDComponents.NDComponent import NDComponent
from neurokernel.LPU.utils.simpleio import *

from collections import OrderedDict

import numpy as np

import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


class BaseAxonHillockModel(with_metaclass(ABCMeta, NDComponent)):
    # __metaclass__ = ABCMeta

    accesses = ['I']
    updates = ['spike_state','V']
    def __init__(self, params_dict, access_buffers, dt,
                 debug=False, LPU_id=None, cuda_verbose=True):
        if cuda_verbose:
            self.compile_options = ['--ptxas-options=-v']
        else:
            self.compile_options = []

        self.num_comps = params_dict[self.params[0]].size
        self.params_dict = params_dict
        self.access_buffers = access_buffers

        self.debug = debug
        self.LPU_id = LPU_id
        self.dtype = params_dict[self.params[0]].dtype

        self.dt = np.double(dt)
        self.ddt = np.double(1e-6)
        self.steps = np.int32(max(int(self.dt/self.ddt), 1))
        print('DEBUG: dt={}, ddt={}'.format(self.dt, self.ddt))
        print('DEBUG: self.steps={}'.format(self.steps))
        # import pdb; pdb.set_trace()
        self.internal_states = {
            c: garray.zeros(self.num_comps, dtype = self.dtype)+self.internals[c] \
            for c in self.internals}

        self.inputs = {
            k: garray.empty(self.num_comps, dtype = self.access_buffers[k].dtype)\
            for k in self.accesses}

        dtypes = {'dt': self.dtype}
        dtypes.update({k: self.inputs[k].dtype for k in self.accesses})
        dtypes.update({k: self.params_dict[k].dtype for k in self.params})
        dtypes.update({k: self.internal_states[k].dtype for k in self.internals})
        dtypes.update({k: self.dtype for k in self.updates})
        self.update_func = self.get_update_func(dtypes)

    def pre_run(self, update_pointers):
        if 'initV' in self.params_dict:
            cuda.memcpy_dtod(int(update_pointers['V']),
                             self.params_dict['initV'].gpudata,
                             self.params_dict['initV'].nbytes)
            cuda.memcpy_dtod(self.internal_states['internalV'].gpudata,
                             self.params_dict['initV'].gpudata,
                             self.params_dict['initV'].nbytes)


    def run_step(self, update_pointers, st=None):
        for k in self.inputs:
            self.sum_in_variable(k, self.inputs[k], st=st)

        self.update_func.prepared_async_call(
            self.update_func.grid, self.update_func.block, st,
            self.num_comps, self.ddt, self.steps,
            *[self.inputs[k].gpudata for k in self.accesses]+\
            [self.params_dict[k].gpudata for k in self.params]+\
            [self.internal_states[k].gpudata for k in self.internals]+\
            [update_pointers[k] for k in self.updates])

    def get_update_template(self):
        template = ''
        return template

    def get_update_func(self, dtypes):
        type_dict = {k: dtype_to_ctype(dtypes[k]) for k in dtypes}
        type_dict.update({'fletter': 'f' if type_dict[self.params[0]] == 'float' else ''})

        # print(type_dict)
        # type_dict = {'dt': 'float', 'I': 'float', 'g_K': 'float', 'g_Na': 'float', 'g_L': 'float', 'E_K': 'float', 'E_Na':
        # 'float', 'E_L': 'float', 'internalVprev1': 'float', 'internalVprev2': 'float', 
        # 'n': 'float', 'm': 'float', 'h': 'float','spike_state': 'float', 'V':'float', 'fletter': 'f'} 
        # print("debug: ", type_dict[self.params[0]] == 'float')
        
        mod = SourceModule(self.get_update_template() % type_dict,
                           options=self.compile_options)
        func = mod.get_function("update")
        func.prepare('i'+np.dtype(dtypes['dt']).char+'i'+'P'*(len(type_dict)-2))
        func.block = (128,1,1)
        func.grid = (min(6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT,
                         (self.num_comps-1) // 128 + 1), 1)
        return func

    def add_initializer(self, var_a, var_b, update_pointers):
        if var_a in self.params_dict:
            if var_b in self.internal_states:
                cuda.memcpy_dtod(self.internal_states[var_b].gpudata,
                                    self.params_dict[var_a].gpudata,
                                    self.params_dict[var_a].nbytes)
            if var_b in update_pointers:
                cuda.memcpy_dtod(int(update_pointers[var_b]),
                                    self.params_dict[var_a].gpudata,
                                    self.params_dict[var_a].nbytes)