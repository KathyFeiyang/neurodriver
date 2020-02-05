
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

dt = 1e-5
dur = 1.0
steps = int(dur/dt)

parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=False,
                    dest='debug', action='store_true',
                    help='Write connectivity structures and inter-LPU routed data in debug folder')
parser.add_argument('-l', '--log', default='none', type=str,
                    help='Log output to screen [file, screen, both, or none; default:none]')
parser.add_argument('-s', '--steps', default=steps, type=int,
                    help='Number of steps [default: %s]' % steps)
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

man = core.Manager()

G = nx.MultiDiGraph()

N = 1024

# Create N HodgkinHuxley2 neurons
for i in range(N):
    id = 'neuron_{}'.format(i)

    G.add_node(id,
               **{'class': 'HodgkinHuxley2',
                  'name': 'HodgkinHuxley2',
                  'g_K': 36.0,
                  'g_Na': 120.0,
                  'g_L': 0.3,
                  'E_K': -77.0,
                  'E_Na': 50.0,
                  'E_L': -54.387
                  })

spk_out_id = 0
in_port_idx = 0

# Create AlphaSynapse connection between each pair of HodgkinHuxley neurons
for i in range(N):
    for j in range(N):
        if i == j:
            continue

        id_i = 'neuron_{}'.format(i)
        id_j = 'neuron_{}'.format(j)
        pair_id = id_i + '_to_' + id_j

        # Projection neurons are all assumed to be attached to output
        # ports (which are represented as separate nodes):
        G.add_node(pair_id + '_out_port',
               **{'class': 'Port',
                  'name': pair_id + '_out_port',
                  'port_type': 'spike',
                  'port_io': 'out',
                  'selector': '/%s/out/spk/%s' % (pair_id, str(spk_out_id))
                 })
        spk_out_id += 1

        G.add_edge(id_i, pair_id + '_out_port')

        # An input port node is created for and attached to each non-projection
        # neuron with a synapse; this assumes that data propagates from one LPU to
        # another as follows:
        # LPU0[projection neuron] -> LPU0[output port] -> LPU1[input port] ->
        # LPU1[synapse] -> LPU1[non-projection neuron]
        G.add_node('in_port' + str(in_port_idx),
                **{'class': 'Port',
                   'name': 'in_port' + str(in_port_idx),
                   'port_type': 'spike',
                   'port_io': 'in',
                   'selector': '/%s/in/spk/%s' % (pair_id, in_port_idx)
                   })

        G.add_edge(pair_id + '_out_port', 'in_port' + str(in_port_idx))

        G.add_node('synapse_' + 'in_port' + str(in_port_idx) + '_' + pair_id,
                **{'class': 'AlphaSynapse',
                   'name': 'in_port' + str(in_port_idx) + '_' + pair_id,
                   'ad': 0.19,
                   'ar': 0.11,
                   'gmax': 0.003,
                   'reverse': 65.0,
                   'circuit': 'local'
                   })

        G.add_edge('in_port' + str(in_port_idx),
                   'synapse_' + 'in_port' + str(in_port_idx) + '_' + pair_id)

        G.add_edge('synapse_' + 'in_port' + str(in_port_idx) + '_' + pair_id, id_j)

        in_port_idx += 1


comp_dict, conns = LPU.graph_to_dicts(G, remove_edge_id = True)


fl_input_processor = StepInputProcessor('I', ['neuron_{}'.format(i) for i in range(N)], 20.0, 0.0, dur)
fl_output_processor = [FileOutputProcessor([('spike_state', None), ('V', None)], 'output.h5', sample_interval=1)]

#fl_output_processor = [OutputRecorder([('spike_state', None), ('V', None)], dur, dt, sample_interval = 1)]

man.add(LPU, 'ge', dt, comp_dict, conns,
        device=args.gpu_dev, input_processors=[fl_input_processor],
        output_processors=fl_output_processor, debug=args.debug,
        print_timing = True, time_sync = False,
        extra_comps=[])

man.spawn()
man.start(steps=args.steps)
start_time = time.time()
man.wait()
end_time = time.time()
print(end_time-start_time)

