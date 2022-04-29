import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from imitation.train_bc import run_task



@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--run_name', type=str, default='')
@click.option('--actor_type', type=str, default='Point')
@click.option('--debug/--no-debug', default=True)
@click.option('--use_pcl', type=str, default='partial_pcl')
@click.option('--eval_train/--no-eval_train', default=True)
@click.option('--dry/--no-dry', default=False)
@click.option('--env', type=str, default='RollExp-v1')
def main(mode, debug, dry, use_pcl, run_name, actor_type, env, eval_train):
    exp_prefix = '1129_Roll_BC_Image'
    vg = VariantGenerator()
    vg.add('algo', ['imitation'])
    vg.add('actor_type', [actor_type])
    vg.add('env_name', [env])
    vg.add("buffer_horizon", [170])
    vg.add('hindsight_goal_ratio', [0.])
    vg.add('debug', [debug])
    vg.add('mid_thres', [0.15])
    vg.add('eval_train', [eval_train])
    vg.add('dataset_path', ['datasets/1129_Roll_exp_gendemo_2021_12_04_20_27_28_0001'])
    vg.add('seed', [100])
    vg.add('open_loop', [False])

    #########
    # for pcl
    if use_pcl:
        # exp_prefix = '0227_BC_video'
        exp_prefix = '1129_Roll_IBC_Point'
        vg.add('use_pcl', [use_pcl])
        vg.add('gt_tool', [True])
        vg.add('rm_eval_noise', [True])
        vg.add('frame', ['world'])
        # vg.add('resume_path', [''])
        
    
    ########
    # to tune
    vg.add('n_counter_example', [256])
    vg.add('temp', [1.])
    vg.add('obs_noise', [0.01])
    vg.add('traj_limit', [-1])
    # vg.add("actor_feature_dim", [1024])
    vg.add('batch_size', [10]) #10
    # vg.add("step_per_epoch", [1024])
    vg.add("il_lr", [1e-4])
    vg.add("frame_stack", [1])
    vg.add('run_name', [run_name])
    
    if debug:
        exp_prefix += '_debug'
        vg.add('il_num_epoch', [5000])
        vg.add('il_eval_freq', [50])
    else:
        vg.add('il_num_epoch', [5000])
        vg.add('il_eval_freq', [50])

    vg.add('exp_prefix', [exp_prefix])
    print('Number of configurations: ', len(vg.variants()))
    sub_process_popens = []
    for idx, vv in enumerate(vg.variants()):
        while len(sub_process_popens) >= 1:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
        compile_script = wait_compile = None

        cur_popen = run_experiment_lite(
            stub_method_call=run_task,
            variant=vv,
            mode=mode,
            dry=dry,
            use_gpu=True,
            exp_prefix=exp_prefix,
            wait_subprocess=debug,
            compile_script=compile_script,
            wait_compile=wait_compile,
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break


if __name__ == '__main__':
    main()
