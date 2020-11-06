from neurokernel.LPU.NDComponents.AxonHillockModels.BaseAxonHillockModel import *

class HodgkinHuxley2(BaseAxonHillockModel):
    updates = ['spike_state', # (bool)
               'V' # Membrane Potential (mV)
              ]
    accesses = ['I'] # Current (\mu A/cm^2)
    params = ['g_K',
              'g_Na',
              'g_L',
              'E_K',
              'E_Na',
              'E_L'
              ]
    internals = OrderedDict([('internalVprev1', 0.),  # Membrane Potential (mV)
                             ('internalVprev2', -65.0),
                             ('n', 0.5),
                             ('m', 0.5),
                             ('h', 0.06)])  # Membrane Potential (mV)]) 
    # extra_params = ['initV',
    #                 'initn',
    #                 'initm',
    #                 'inith']

    @property
    def maximum_dt_allowed(self):
        return 1e-3

    # def pre_run(self, update_pointers):
    #     super(HodgkinHuxley2, self).pre_run(update_pointers)
    #     self.add_initializer('initV', 'Vprev1', update_pointers)
    #     self.add_initializer('initn', 'n', update_pointers)
    #     self.add_initializer('initm', 'm', update_pointers)
    #     self.add_initializer('inith', 'h', update_pointers)

    def get_update_template(self):
        template = """
#define EXP exp%(fletter)s
#define POW pow%(fletter)s
#define ABS fabs%(fletter)s
#define LOG log%(fletter)s

__global__ void update(
    int num_comps,
    %(dt)s dt,
    int nsteps,
    %(input_I)s* g_I,
    %(param_g_K)s* g_g_K,
    %(param_g_Na)s* g_g_Na,
    %(param_g_L)s* g_g_L,
    %(param_E_K)s* g_E_K,
    %(param_E_Na)s* g_E_Na,
    %(param_E_L)s* g_E_L,
    %(internal_internalVprev1)s* g_internalVprev1,
    %(internal_internalVprev2)s* g_internalVprev2,
    %(internal_n)s* g_n,
    %(internal_m)s* g_m,
    %(internal_h)s* g_h,
    %(update_spike_state)s* g_spike_state,
    %(update_V)s* g_V)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    // %(dt)s ddt = dt*1000.; // s to ms

    %(update_V)s V, Vprev1, Vprev2;
    %(input_I)s I, I_K, I_Na, I_L, I_channels;
    %(update_spike_state)s spike_state;
    %(param_g_Na)s g_Na;
    %(param_g_K)s g_K;
    %(param_g_L)s g_L;

    %(param_E_Na)s E_Na;
    %(param_E_K)s E_K;
    %(param_E_L)s E_L;

    %(internal_n)s n, a_n, b_n;
    %(internal_m)s m, a_m, b_m;
    %(internal_h)s h, a_h, b_h;

    for(int i = tid; i < num_comps; i += total_threads)
    {
        V = g_internalVprev1[i];
        Vprev1 = g_internalVprev1[i];
        Vprev2 = g_internalVprev2[i];
        m = g_m[i];
        n = g_n[i];
        h = g_h[i];
        I = g_I[i];
        spike_state = 0;
        E_Na = g_E_Na[i];
        E_K = g_E_K[i];
        E_L = g_E_L[i];
        g_Na = g_g_Na[i];
        g_K = g_g_K[i];
        g_L = g_g_L[i];

        for (int j = 0; j < nsteps; ++j)
        {
            a_m = (25.0 - V) / (10.0 * (EXP((25.0 - V) / 10.0) - 1.0));
            a_n = (10.0 - V) / (100.0 * (EXP((10.0 - V) / 10.0) - 1.0));
            a_h = 0.07 * EXP(-V / 20.0);

            b_m = 4.0 * EXP(-1.0 * V / 18.0);
            b_n = 0.125 * EXP(-1.0 * V / 80.0);
            b_h = 1.0 / (EXP((30.0 - V) / 10.0) + 1.0);
            
            m = m + dt * (a_m * (1.0 - m) - b_m * m);
            n = n + dt * (a_n * (1.0 - n) - b_n * n);
            h = h + dt * (a_h * (1.0 - h) - b_h * h);
            
            I_K = g_K * EXP(4 * LOG(n)) * (V - E_K);
            I_Na = g_Na * EXP(3 * LOG(m)) * h * (V - E_Na);
            I_L = g_L * (V - E_L);
            I_channels = I_K + I_Na + I_L;

            V = V + dt * (I - I_channels);
            
            spike_state += (Vprev2 <= Vprev1) && (Vprev1 >= V) && (V > -30);
            
            Vprev2 = Vprev1;
            Vprev1 = V;
        }
        g_V[i] = V;
        g_internalVprev1[i] = Vprev1;
        g_internalVprev2[i] = Vprev2;
        g_m[i] = m;
        g_n[i] = n;
        g_h[i] = h;
        g_spike_state[i] = (spike_state > 0);       
    }
}
"""
        return template


if __name__ == '__main__':
    import argparse
    import itertools
    import networkx as nx
    from neurokernel.tools.logging import setup_logger
    import neurokernel.core_gpu as core

    from neurokernel.LPU.LPU import LPU

    from neurokernel.LPU.InputProcessors.FileInputProcessor import FileInputProcessor
    from neurokernel.LPU.InputProcessors.StepInputProcessor import StepInputProcessor
    from neurokernel.LPU.OutputProcessors.FileOutputProcessor import FileOutputProcessor

    import neurokernel.mpi_relaunch

    dt = 1e-6
    dur = 1e-5
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

    G.add_node('neuron0', **{
               'class': 'HodgkinHuxley2',
               'name': 'HodgkinHuxley2',
               'g_K': 36.0,
               'g_Na': 120.0,
               'g_L': 0.3,
               'E_K': -77.0,
               'E_Na': 50.0,
               'E_L': -54.387,
               'spike_state': 0,
               'V': 0
               })

    comp_dict, conns = LPU.graph_to_dicts(G)

    fl_input_processor = StepInputProcessor('I', ['neuron0'], 20, 0, 1e-5)
    fl_output_processor = FileOutputProcessor([('spike_state', None), ('V', None)], 'new_output.h5', sample_interval=1)

    man.add(LPU, 'ge', dt, comp_dict, conns,
            device=args.gpu_dev, input_processors = [fl_input_processor],
            output_processors = [fl_output_processor], debug=args.debug)

    man.spawn()
    man.start(steps=args.steps)
    man.wait()

    # plot the result
    import h5py
    # import matplotlib
    # matplotlib.use('PS')
    # import matplotlib.pyplot as plt

    f = h5py.File('new_output.h5')
    t = np.arange(0, args.steps)*dt
    print(f.keys())
    print(np.array(list(f['V'].values())[0]))
    # print(np.array(list(f['V'].values())[1]))
    print(np.array(list(f['spike_state'].values())[0]))

    # plt.figure()
    # plt.plot(t,list(f['V'].values())[0])
    # plt.xlabel('time, [s]')
    # plt.ylabel('Voltage, [mV]')
    # plt.title('Hodgkin-Huxley Neuron')
    # plt.savefig('hhn.png',dpi=300)
