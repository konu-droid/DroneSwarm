from time import sleep
import os

# create isaac environment
from omni.isaac.gym.vec_env import VecEnvBase

env = VecEnvBase(headless=False)
# when you need all the extensions of isaac sim use the below line
#  env = VecEnvBase(headless=False, experience=f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit')

from arm_task import ArmTask

task = ArmTask(name="Fly")
env.set_task(task, backend="torch")
# check_env(env)

env._world.reset()
obs, _ = env.reset()
temp = 0.07
actions = [temp, temp, temp ,temp]
while env._simulation_app.is_running():
    env.step(actions)
    # sleep(0.05)

env.close()
