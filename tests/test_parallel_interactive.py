from nested.utils import Context, get_unknown_click_arg_dict
from nested.parallel import get_parallel_interface
from nested.optimize_utils import nested_parallel_init_contexts_interactive
import click
import sys
import os


context = Context()


def config_controller():
    if 'controller_comm' in context():
        if context.disp:
            print('context.controller_comm is defined on controller with size: %i' % context.controller_comm.size)
            sys.stdout.flush()
    else:
        raise RuntimeError('config_controller: context.controller_comm is not defined')


def config_worker():
    context.updated = False
    if 'comm' in context():
        if context.disp:
            print('pid: %i; context.comm is defined on worker with size: %i' % (os.getpid(), context.comm.size))
            sys.stdout.flush()
    else:
        raise RuntimeError('config_worker: context.comm is not defined on a worker')

@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True,))
@click.option("--config-file-path", type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option("--output-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default='data')
@click.option("--label", type=str, default=None)
@click.option("--disp", is_flag=True)
@click.option("--interactive", is_flag=True)
@click.option("--framework", type=str, default='serial')
@click.pass_context
def main(cli, config_file_path, output_dir, label, disp, interactive, framework):
    """
    :param cli: :class:'click.Context': used to process/pass through unknown click arguments
    :param config_file_path: str (path)
    :param output_dir: str
    :param label: str
    :param disp: bool
    :param interactive: bool
    :param framework: str
    """
    kwargs = get_unknown_click_arg_dict(cli.args)
    context.interface = get_parallel_interface(framework, **kwargs)
    context.interface.start(disp=disp)
    context.interface.ensure_controller()
    nested_parallel_init_contexts_interactive(context, config_file_path, label, output_dir, disp, **kwargs)

    if not interactive:
        context.interface.stop()


if __name__ == '__main__':
    main(standalone_mode=False)