
import sys
import time
import argparse
import itertools
import dill
import pickle
import functools
import uuid
import networkx as nx
import numpy as np
from neurokernel.tools.logging import setup_logger

from neurokernel.LPU.LPU import LPU

from neurokernel.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
from neurokernel.LPU.InputProcessors.StepInputProcessor import StepInputProcessor
from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor
from neurokernel.LPU.OutputProcessors.OutputRecorder import OutputRecorder

parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=False,
                    dest='debug', action='store_true',
                    help='Write connectivity structures and inter-LPU routed data in debug folder')
parser.add_argument('-l', '--log', default='none', type=str,
                    help='Log output to screen [file, screen, both, or none; default:none]')
# parser.add_argument('-s', '--steps', default=steps, type=int,
#                     help='Number of steps [default: %s]' % steps)
parser.add_argument('-g', '--gpu_dev', default=0, type=int,
                    help='GPU device number [default: 0]')
args = parser.parse_args()

file_name = None
screen = False
if args.log.lower() in ['file', 'both']:
    file_name = 'neurokernel_{}.log'.format(str(uuid.uuid4())[:4])
if args.log.lower() in ['screen', 'both']:
    screen = True
logger = setup_logger(file_name=file_name, screen=screen)
print('Logging in file {}'.format(file_name))

def create_graph(N):
    G = nx.MultiDiGraph()

    final_neuron_id = "neuron_final"
    G.add_node(final_neuron_id,
            **{'class': 'HodgkinHuxley2',
                'name': 'HodgkinHuxley2',
                'g_K': 36.0,
                'g_Na': 120.0,
                'g_L': 0.3,
                'E_K': -12.0,
                'E_Na': 115.0,
                'E_L': 10.6,
                'V': 0.0,
                'spike_state': 0,
                })
    # Create N HodgkinHuxley2 neurons and N AlphaSynapse connections
    for i in range(N):
        neuron_id = 'neuron_{}'.format(i)
        G.add_node(neuron_id,
                **{'class': 'HodgkinHuxley2',
                    'name': 'HodgkinHuxley2',
                    'g_K': 36.0,
                    'g_Na': 120.0,
                    'g_L': 0.3,
                    'E_K': -12.0,
                    'E_Na': 115.0,
                    'E_L': 10.6,
                    'V': 0.0,
                    'spike_state': 0,
                    })

        synapse_id = 'synapse_{}'.format(i)
        G.add_node(synapse_id,
                **{'class': 'GABA_B',
                    'name': 'GABA_B',
                    'gmax': 1e7,
                    'n': 4,
                    'gamma': 1.0,
                    'alpha_1': 5,
                    'alpha_2': 0.018,
                    'beta_1': 0.12,
                    'beta_2': 0.034,
                    'reverse': 100.0
                    })

        G.add_edge(neuron_id, synapse_id)
        G.add_edge(synapse_id, final_neuron_id)

    return G


def simulation(dt, N, output_n):
    start_time = time.time()

    dur = 50
    steps = int(np.round(dur/dt))

    G = create_graph(N)
    print("Creating graph completed in {} seconds.".format(time.time()-start_time))

    compile_start_time = time.time()
    #comp_dict, conns = LPU.graph_to_dicts(G, remove_edge_id=False)

    fl_input_processor = StepInputProcessor('I', ['neuron_{}'.format(i) for i in range(N)], 100.0, 0.0, dur)
    # fl_output_processor = [FileOutputProcessor([('V', None), ('g', ['synapse_neuron_{}_to_neuron_1'.format(i) for i in range(N)])],# ('spike_state', None), ('g', None), ('E', None)],
    #                                           'neurodriver_output_{}.h5'.format(output_n), sample_interval=1, cache_length=2000)]
    fl_output_processor = FileOutputProcessor([('spike_state', None), ('V', None), ('g', None)], 'hh_gaba-b_neurodriver.h5', sample_interval=1)
    # fl_output_processor = [] # temporarily suppress generating output

    #fl_output_processor = [OutputRecorder([('spike_state', None), ('V', None), ('g', None), ('E', None)], dur, dt, sample_interval = 1)]

    lpu = LPU(dt, 'obj', G,
              device=args.gpu_dev, id = 'ge', input_processors=[fl_input_processor],
              output_processors=fl_output_processor, debug=args.debug, manager = False,
              print_timing=False, time_sync=False, extra_comps=[])
    print("Instantiating LPU completed in {} seconds.".format(time.time()-start_time))
    execute_start_time = time.time()
    # LPU.run includes pre_run, run_steps and post_run
    lpu.run(steps = steps)
    execution_time = time.time() - execute_start_time
    compile_and_execute_time = time.time() - compile_start_time
    print("LPUs Compilation and Execution Completed in {} seconds.".format(compile_and_execute_time))
    return compile_and_execute_time, execution_time

if __name__ == '__main__':
    sim_time = []
    compile_and_sim_time = []

    #diff_dt = [5e-6, 1e-5, 5e-5, 1e-4, 1e-3]
    #diff_N = [2, 32, 128, 256, 512]
    diff_dt = [1e-3] # change ddt accordingly
    diff_N = [4]#[2, 32, 128, 256, 512]
    n_sim = 1
    i = 0
    for dt in diff_dt:
        for N in diff_N:
            sim_time.append([])
            compile_and_sim_time.append([])
            for t in range(n_sim + 1):
                c, s = simulation(dt, N, i * n_sim + t)
                sim_time[i].append(s)
                compile_and_sim_time[i].append(c)
            sim_time[i].pop(0) # discard first result
            compile_and_sim_time[i].pop(0)
            i += 1
    sim_averages = [sum(result) / n_sim for result in sim_time]
    compile_and_sim_averages = [sum(result) / n_sim for result in compile_and_sim_time]
    print("==========================================")
    print("diff_N:", diff_N)
    print("diff_dt:", diff_dt)
    print("n_sim:", n_sim)
    print("Simulation results:\n", sim_time, "\n", sim_averages)
    print("Compilation and Simulation results:\n", compile_and_sim_time, "\n", compile_and_sim_averages)
    print("==========================================")

    import h5py
    f = h5py.File('hh_gaba-b_neurodriver.h5')
    print(f.keys())
    print(np.array(list(f['V'].values())[0]))
    print(np.array(list(f['g'].values())[0]))
