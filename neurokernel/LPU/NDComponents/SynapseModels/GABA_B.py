from neurokernel.LPU.NDComponents.SynapseModels.BaseSynapseModel import *

class GABA_B(BaseSynapseModel):
    accesses = ['spike_state'] # (bool)
    updates = ['g'] # conductance (mS/cm^2)
    params = ['gmax',
              'alpha_1', # maximum conductance (mS/cm^2)
              'alpha_2', # rise rate of conductance (ms)
              'beta_1', # decay rate of conductance (ms)
              'beta_2', # decay rate of conductance (ms)
              'n', # decay rate of conductance (ms)
              'gamma', # decay rate of conductance (ms)
              ]
    internals = OrderedDict([('x1', 0.0),  # g,
                             ('x2', 0.0),  # derivative of g
                             ])

    @property
    def maximum_dt_allowed(self):
        return 1e-3

    def get_update_template(self):
        template = """
__global__ void update(int num_comps, %(dt)s dt, int steps,
                       %(input_spike_state)s* g_spike_state,
                       %(param_gmax)s* g_gmax, 
                       %(param_alpha_1)s* g_alpha_1, %(param_alpha_2)s* g_alpha_2,
                       %(param_beta_1)s* g_beta_1, %(param_beta_2)s* g_beta_2,
                       %(param_n)s* g_n, %(param_gamma)s* g_gamma,
                       %(internal_x1)s* g_x1, %(internal_x2)s* g_x2,
                       %(update_g)s* g_g)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    // %(dt)s ddt = dt*1000.; // s to ms
    %(input_spike_state)s spike_state;
    %(param_gmax)s gmax;
    %(param_n)s n;
    %(param_gamma)s gamma;
    %(param_alpha_1)s alpha_1;
    %(param_alpha_2)s alpha_2;
    %(param_beta_1)s beta_1;
    %(param_beta_2)s beta_2;
    %(internal_x1)s x1;
    %(internal_x2)s x2;

    for(int i = tid; i < num_comps; i += total_threads)
    {
        alpha_1 = g_alpha_1[i];
        alpha_2 = g_alpha_2[i];
        beta_1 = g_beta_1[i];
        beta_2 = g_beta_2[i];
        n = g_n[i];
        gamma = g_gamma[i];
        x1 = g_x1[i];
        x2 = g_x2[i];
        spike_state = g_spike_state[i];

        x1 = x1 + dt * ( alpha_1 * (spike_state) * (1 - x1) - beta_1 * x1);
        x2 = x2 + dt * (alpha_2 * x1 - beta_2 * x2);

        gmax = g_gmax[i];
        g_x1[i] = x1;
        g_x2[i] = x2;
        g_g[i] = gmax * (powf(x1, n) / (gamma + powf(x2, n)));
    }
}
"""
        return template

if __name__ == '__main__':
    import argparse
    import itertools

    import networkx as nx
    import h5py

    from neurokernel.tools.logging import setup_logger
    import neurokernel.core_gpu as core
    from neurokernel.LPU.LPU import LPU
    from neurokernel.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
    from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor
    import neurokernel.mpi_relaunch

    dt = 1e-4
    dur = 1.0
    steps = int(dur / dt)

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', default=False,
                        dest='debug', action='store_true',
                        help='Write connectivity structures and inter-LPU routed data in debug folder')
    parser.add_argument('-l', '--log', default='both', type=str,
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

    t = np.arange(0, dt * steps, dt)

    uids = np.array(["synapse0"], dtype='S')

    spike_state = np.zeros((steps, 1), dtype=np.int32)
    spike_state[np.nonzero((t - np.round(t / 0.04) * 0.04) == 0)[0]] = 1

    with h5py.File('input_spike.h5', 'w') as f:
        f.create_dataset('spike_state/uids', data=uids)
        f.create_dataset('spike_state/data', (steps, 1),
                         dtype=np.int32,
                         data=spike_state)

    man = core.Manager()

    G = nx.MultiDiGraph()

    G.add_node('synapse0', **{
               'class': 'GABA_B',
               'name': 'GABA_B',
               'gmax': 0.003 * 1e-3,
               'n': 4,
               'gamma': 100,
               'alpha_1': 0.09,
               'alpha_2': 0.18,
               'beta_1': 0.0012,
               'beta_2': 0.034,
               'reverse': -95.0
               })

    comp_dict, conns = LPU.graph_to_dicts(G)

    fl_input_processor = FileInputProcessor('input_spike.h5')
    fl_output_processor = FileOutputProcessor(
        [('g', None)], 'new_output.h5', sample_interval=1)

    man.add(LPU, 'ge', dt, comp_dict, conns,
            device=args.gpu_dev, input_processors=[fl_input_processor],
            output_processors=[fl_output_processor], debug=args.debug)

    man.spawn()
    man.start(steps=args.steps)
    man.wait()
