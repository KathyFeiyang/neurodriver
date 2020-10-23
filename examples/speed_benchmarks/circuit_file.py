
import time
import argparse
import itertools
import networkx as nx
from neurokernel.tools.logging import setup_logger
import neurokernel.core_gpu as core

from neurokernel.LPU.LPU import LPU

from neurokernel.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
from neurokernel.LPU.InputProcessors.StepInputProcessor import StepInputProcessor
from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor
from neurokernel.LPU.OutputProcessors.OutputRecorder import OutputRecorder

import neurokernel.mpi_relaunch


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
    file_name = 'neurokernel.log'
if args.log.lower() in ['screen', 'both']:
    screen = True
logger = setup_logger(file_name=file_name, screen=screen)


def simulation(dt, N, output_n):
    start_time = time.time()

    #dt = 1e-3
    dur = 1.0 / 100
    steps = int(dur/dt)

    man = core.Manager()

    G = nx.MultiDiGraph()

    # Create N HodgkinHuxley2 neurons
    for i in range(N):
        id = 'neuron_{}'.format(i)

        G.add_node(id, **{
            'class': 'HodgkinHuxley2',
            'name': 'HodgkinHuxley2',
            'g_K': 36.0,
            'g_Na': 120.0,
            'g_L': 0.3,
            'E_K': -77.0,
            'E_Na': 50.0,
            'E_L': -54.387,
            'spike': 0,
            'V': 0.0
            })

    spk_out_id = 0
    in_port_idx = 0

    # Create Hodgkin Huxley connections to Alpha Synapses
    # Create AlphaSynapse --> Aggregator (Dendrite) --> LeakyIAF connections
    for i in range(N):
        for j in range(N):
            id_i = 'neuron_{}'.format(i)
            id_j = 'neuron_{}'.format(j)
            pair_id = id_i + '_to_' + id_j
            synapse_id = 'synapse_' + pair_id
            G.add_node(synapse_id, **{
                'class': 'AlphaSynapse',
                'name': 'AlphaSynapse',
                'ar': 4.0,
                'ad': 4.0,
                'reverse': 100.0,
                'gmax': 100.0,
                'g': 0.0,
                'E': 0.0,
                'circuit': 'local'})
            G.add_edge(id_i, synapse_id)
            
            aggregator_id = 'aggregator_' + synapse_id
            G.add_node(aggregator_id, **{
                'class': 'Aggregator',
                'name': aggregator_id,
                'I': 0.0})
            G.add_edge(synapse_id, aggregator_id)

            leakyIAF_id = 'leakyIAF_' + aggregator_id
            G.add_node(leakyIAF_id, **{
                'class': 'LeakyIAF',
                'name': leakyIAF_id,
                'resting_potential': -70.0,
                'threshold': -40.0,
                'reset_potential': -70.0,
                'capacitance': 1.0,
                'resistance': 0.007,
                'spike_state': 0,
                'V': 0.0})
            G.add_edge(aggregator_id, leakyIAF_id)

    comp_dict, conns = LPU.graph_to_dicts(G, remove_edge_id=False)

    fl_input_processor = StepInputProcessor('I', ['neuron_{}'.format(i) for i in range(N)], 20.0, 0.0, dur)
    fl_output_processor = [FileOutputProcessor([('V', None), ('spike_state', None), ('g', None), ('E', None), ('I', None)], 'neurodriver_output.h5'.format(output_n), sample_interval=1, cache_length=2000)]
    #fl_output_processor = [] # temporarily suppress generating output

    #fl_output_processor = [OutputRecorder([('spike_state', None), ('V', None)], dur, dt, sample_interval = 1)]

    man.add(LPU, 'ge', dt, comp_dict, conns,
            device=args.gpu_dev, input_processors=[fl_input_processor],
            output_processors=fl_output_processor, debug=args.debug,
            print_timing=False, time_sync=False,
            extra_comps=[])

    man.spawn()
    print('DEBUG: #simulation steps={}'.format(steps))
    man.start(steps=steps)
    man.wait()
    end_time = time.time()
    sim_time = end_time - start_time
    print(end_time-start_time)

    return sim_time

if __name__ == '__main__':
    results = []
    #diff_dt = [5e-6, 1e-5, 5e-5, 1e-4, 1e-3]
    #diff_N = [2, 32, 128, 256, 512]
    diff_dt = [1e-7] # remember change ddt accordingly in LPU/NDComponents/AxonHillockModels/BaseAxonHillockModel.py
    diff_N = [2, 128, 512]
    n_sim = 1
    i = 0
    for dt in diff_dt:
        for N in diff_N:
            results.append([])
            for t in range(n_sim + 1):
                results[i].append(simulation(dt, N, i * n_sim + t))
            results[i] = results[i][1:] # discard first result
            i += 1
    averages = [sum(result) / n_sim for result in results]
    print("==========================================")
    print("diff_N:", diff_N)
    print("diff_dt:", diff_dt)
    print("n_sim:", n_sim)
    print("Simulation results:\n", results, "\n", averages)
    print("==========================================")
