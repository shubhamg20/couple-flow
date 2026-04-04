import argparse

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# (Optional) keep the app alive
while simulation_app.is_running():
    simulation_app.update()
"""Rest everything follows."""