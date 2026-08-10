import h5py
import os


def merge_timeframes(results_path, out_path):

    # Get list of timeframe directories
    timeframe_dirs = sorted([d for d in os.listdir(results_path) if ('timeframe' in d)])

    # Load velocities from the first timeframe
    with h5py.File(os.path.join(results_path, timeframe_dirs[0], 'pred.h5'), 'r') as f:
        u = f['u'][:]
        v = f['v'][:]
        w = f['w'][:]

    # Determine the shape of the time resolved velocities
    num_timeframes = len(timeframe_dirs)
    u_time_resolved = np.zeros((num_timeframes, *u.shape))
    v_time_resolved = np.zeros((num_timeframes, *v.shape))
    w_time_resolved = np.zeros((num_timeframes, *w.shape))
    print(f"Time resolved velocities shape: {u_time_resolved.shape}")
    exit()

    # Assign the first timeframe
    u_time_resolved[0] = u
    v_time_resolved[0] = v
    w_time_resolved[0] = w

    # Load and assign the rest of the timeframes
    for i, timeframe_dir in enumerate(timeframe_dirs[1:], start=1):
        with h5py.File(os.path.join(results_path, timeframe_dir, 'pred.h5'), 'r') as f:
            u_time_resolved[i] = f['u'][:]
            v_time_resolved[i] = f['v'][:]
            w_time_resolved[i] = f['w'][:]

    # Save the time resolved velocities
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('u', data=u_time_resolved)
        f.create_dataset('v', data=v_time_resolved)
        f.create_dataset('w', data=w_time_resolved)
    

if __name__ == '__main__':
    results_path = "/proj/multipress/users/x_javbi/SRFLOW2/SRFlowNIR/models/250213_00100833_HNCM_V150_all_timeframes_fs125"
    out_path = "/proj/multipress/users/x_javbi/SRFLOW2/SRFlowNIR/models/250213_00100833_HNCM_V150_all_timeframes_fs125/time_resolved_velocities.h5"
    merge_timeframes(results_path, out_path)

        