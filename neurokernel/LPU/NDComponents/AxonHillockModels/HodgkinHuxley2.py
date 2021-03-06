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
    internals = OrderedDict([('internalV',-65.),       # Membrane Potential (mV)
                             ('internalVprev1',-65.),  # Membrane Potential (mV)
                             ('internalVprev2',-65.),
                             ('n', 0.),
                             ('m', 0.),
                             ('h', 0.92)]) # Membrane Potential (mV)

    def get_update_template(self):
        template = """
#define EXP exp%(fletter)s
#define POW pow%(fletter)s
#define ABS fabs%(fletter)s

__global__ void update(
    int num_comps,
    %(dt)s dt,
    int nsteps,
    %(I)s* g_I,
    %(g_K)s* g_g_K,
    %(g_Na)s* g_g_Na,
    %(g_L)s* g_g_L,
    %(E_K)s* g_E_K,
    %(E_Na)s* g_E_Na,
    %(E_L)s* g_E_L,
    %(internalV)s* g_internalV,
    %(internalVprev1)s* g_internalVprev1,
    %(internalVprev2)s* g_internalVprev2,
    %(n)s* g_n,
    %(m)s* g_m,
    %(h)s* g_h,
    %(spike_state)s* g_spike_state,
    %(V)s* g_V)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;

    %(dt)s ddt = dt*1000.; // s to ms

    %(V)s V, Vprev1, Vprev2, dV;
    %(I)s I;
    %(spike_state)s spike;
    %(g_Na)s g_Na;
    %(g_K)s g_K;
    %(g_L)s g_L;

    %(E_Na)s E_Na;
    %(E_K)s E_K;
    %(E_L)s E_L;

    %(n)s n, dn;
    %(m)s m, dm;
    %(h)s h, dh;
    %(n)s a;

    for(int i = tid; i < num_comps; i += total_threads)
    {
        spike = 0;
        V = g_internalV[i];
        Vprev1 = g_internalVprev1[i];
        Vprev2 = g_internalVprev2[i];
        I = g_I[i];
        n = g_n[i];
        m = g_m[i];
        h = g_h[i];
        g_Na = g_g_Na[i];
        g_K = g_g_K[i];
        g_L = g_g_L[i];
        E_Na = g_E_Na[i];
        E_K = g_E_K[i];
        E_L = g_E_L[i];

        for (int j = 0; j < nsteps; ++j)
        {
            a = exp(-(V+55)/10)-1;
            if (ABS(a) <= 1e-7)
                dn = (1.-n) * 0.1 - n * (0.125*EXP(-(V+65.)/80.));
            else
                dn = (1.-n) * (-0.01*(V+55.)/a) - n * (0.125*EXP(-(V+65)/80));

            a = exp(-(V+40.)/10.)-1.;
            if (ABS(a) <= 1e-7)
                dm = (1.-m) - m*(4*EXP(-(V+65)/18));
            else
                dm = (1.-m) * (-0.1*(V+40.)/a) - m * (4.*EXP(-(V+65.)/18.));

            dh = (1.-h) * (0.07*EXP(-(V+65.)/20.)) - h / (EXP(-(V+35.)/10.)+1.);

            dV = I - g_Na*POW(m,3)*h*(V-E_Na) - g_K * POW(n,4) * (V-E_K) - g_L * (V-E_L);

            n += ddt * dn;
            m += ddt * dm;
            h += ddt * dh;
            V += ddt * dV;

            spike += (Vprev2<=Vprev1) && (Vprev1 >= V) && (Vprev1 > -30);

            Vprev2 = Vprev1;
            Vprev1 = V;
        }

        g_n[i] = n;
        g_m[i] = m;
        g_h[i] = h;
        g_V[i] = V;
        g_internalV[i] = V;
        g_internalVprev1[i] = Vprev1;
        g_internalVprev2[i] = Vprev2;
        g_spike_state[i] = (spike > 0);
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

    dt = 1e-4
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

    G.add_node('neuron0', **{
               'class': 'HodgkinHuxley',
               'name': 'HodgkinHuxley',
               'n': 0.,
               'm': 0.,
               'h': 1.,
               })

    comp_dict, conns = LPU.graph_to_dicts(G)

    fl_input_processor = StepInputProcessor('I', ['neuron0'], 40, 0.2, 0.8)
    fl_output_processor = FileOutputProcessor([('spike_state', None),('V', None)], 'new_output.h5', sample_interval=1)

    man.add(LPU, 'ge', dt, comp_dict, conns,
            device=args.gpu_dev, input_processors = [fl_input_processor],
            output_processors = [fl_output_processor], debug=args.debug)

    man.spawn()
    man.start(steps=args.steps)
    man.wait()

    # plot the result
    import h5py
    import matplotlib
    matplotlib.use('PS')
    import matplotlib.pyplot as plt

    f = h5py.File('new_output.h5')
    t = np.arange(0, args.steps)*dt

    plt.figure()
    plt.plot(t,list(f['V'].values())[0])
    plt.xlabel('time, [s]')
    plt.ylabel('Voltage, [mV]')
    plt.title('Hodgkin-Huxley Neuron')
    plt.savefig('hhn.png',dpi=300)
