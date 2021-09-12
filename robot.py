import pybullet as p
import random, math, time
import pybullet_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statistics import harmonic_mean
from scipy.optimize import minimize, brute
# from sklearn import model_selection#, datasets, linear_model, metrics
# from matplotlib import animation
# from IPython.display import display, HTML
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.optimizers import RMSprop

fpss = 0.0041
secs = 2.2
fps = 240
stop_step = int(240*secs)
PLATFORM_ELEVATION = 0.05
WANTED_HEIGHT = 0.3

class Enviroment:
    def __init__(self, fpss, GUI_enable=True):
        self.GUI_enable = GUI_enable
        if GUI_enable:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        # p.setTimeStep(1 / 240/4)
        p.setPhysicsEngineParameter(enableSAT=1)

        if GUI_enable:
            p.resetDebugVisualizerCamera(1, 52, -32, (-0.26, 0.22, -0.24))
            self.fps = p.addUserDebugParameter("fps", 1, 800, 1)
        else:
            self.fpss = fpss

    def update_fps(self):
        if self.GUI_enable:
            self.fpss = 1 / p.readUserDebugParameter(self.fps)
        else:
            pass


    def start(self, ball_loc, ball_vel, action_params, debug=False):
        # print()
        # print(ball_loc, ball_vel, action_params)
        self.debug = debug
        self.steps = 0
        self.rebounds = 0
        self.ball_moving = True  # False if Z_velosity = 0
        self.ball_falls = True  # False if Z_velosity > 0
        self.hit_counter = 0
        self.aver_z_vel = 0
        self.next_state_cords = 0
        self.next_state_vel = 0
        self.contact_loc = (0, 0)  # last contact location of ball-platform
        self.stop = False
        self.A = self.B = self.C = 0
        self.reward = 0
        self.bad_final = False
        self.action_params = action_params

        self.cubeId = p.loadURDF("mycube.urdf", [0, 0, 0], globalScaling=1, useFixedBase=True, flags=p.URDF_INITIALIZE_SAT_FEATURES)
        self.makeBall(ball_loc)
        self.pushBall(ball_vel)

        self.timer = self.Cheduler()
        self.timer.add_job(self.steps + 1, self.action)

    def makeBall(self, ball_loc):
        self.ballId = p.loadURDF("myball.urdf", [ball_loc[0], ball_loc[1], ball_loc[2]], globalScaling=1, flags=p.URDF_INITIALIZE_SAT_FEATURES)
        p.changeDynamics(self.ballId, -1, mass=1, restitution=0.8)

    def getCords(selfs, id):
        return np.around(p.getBasePositionAndOrientation(id)[0], 5)

    def getVel(selfs, id):
        return p.getBaseVelocity(id)[0]

    # def reaction(self, x, y, x_vel, y_vel, pro, der):
    #     if (self.steps % 1 == 0):
    #         self.move(y*pro+y_vel*der, -(x*pro+x_vel*der))

    # run when hit detected
    def on_hit(self):
        if self.debug:
            self.contact_loc = self.getCords(self.ballId)[:2]
            print('===== HIT FUN', self.contact_loc, )
        # stats.addMarker(self.steps)

        if self.hit_counter == 0:
            platZ = p.getLinkState(self.cubeId, 2)[0][2]
            half = PLATFORM_ELEVATION / 2
            C = -abs(platZ - half) * (1 / half) + 1.001
            self.C = 0 if C < 0 else C
            if self.debug:
                print('\nC = {}      because elevation = {}'.format(self.C, platZ))
            if self.action_params[2] == 9999: # "check mode" in documentation
                self.A, self.B, self.C = 2, 2, 2
                self.final()
        elif self.hit_counter == 1:
            z_vel_platform = p.getLinkState(self.cubeId, 2, 1)[6][2]
            if z_vel_platform < 0 and abs(z_vel_platform) > 0.1:
                self.bad_final = True
            self.contact_loc = self.getCords(self.ballId)[:2]
            x, y = self.contact_loc
            A = ((abs(x) * abs(y) * 1 / 0.15) - (abs(x) + abs(y)) + 0.15) * (1 / 0.15)
            # print('\nA = {}      x,y={} {}'.format(A, x, y))
            self.A = 0 if A < 0 else A
            if self.debug:
                print('\nA = {}      x,y={} {}'.format(self.A, x, y))
            self.final()
        else:
            print('=== ERROR on hit ===')
            self.bad_final = True
            self.final()

        self.hit_counter += 1

    # run when ball begins fall
    def on_fall(self):
        Z = self.getCords(self.ballId)[2]
        B = -abs(Z - WANTED_HEIGHT) * (1 / WANTED_HEIGHT) + 1
        self.B = 0 if B < 0 else B
        if self.debug:
            print('\nB = {}      because max_height = {}'.format(self.B , Z))
            print('===== FALL FUN ===== cords,vel SAVED')

        self.next_state_cords = self.getCords(self.ballId)
        self.next_state_vel = self.getVel(self.ballId)[:2]

    def final(self):
        self.reward = harmonic_mean([self.A, self.B])*0.66 + self.C * 0.34
        if self.bad_final:
            self.reward = 0
        # if self.reward > 0 or self.debug:
        #     print('\nreward={} because A={} B={} C={}'.format(self.reward, self.A, self.B, self.C))
        self.stop = True

    def pushBall(self, ball_vel):
        if self.debug:
            print('\nforce applied: {} {}'.format(ball_vel[0],ball_vel[1]))
        p.applyExternalForce(self.ballId, -1, [ball_vel[0], ball_vel[1], 0], [0,0,0], p.LINK_FRAME)

    def step(self):

        self.update_fps()

        self.steps += 1
        p.stepSimulation()
        self.timer(self.steps)

        X, Y, Z = self.getCords(self.ballId)
        ball_vel = self.getVel(self.ballId)

        self.triggers(ball_vel, (X, Y, Z))
        # platZ = p.getLinkState(0, 2, 1)[6][2]
        platZ = p.getLinkState(self.cubeId, 2)[0][2]

        if (Z < -0.1): # or (abs(X) > 0.2) or (abs(Y) > 0.2)
            self.bad_final = True
            self.stop = True

    def triggers(self, vel, cords):
        vel_z = vel[2]
        alpha = 0.5
        # exponential moving average
        self.aver_z_vel = round(self.aver_z_vel*(1-alpha) + vel_z*alpha, 4)

        self.ball_moving = True if abs(self.aver_z_vel) > 0.1 else False # update moving flag

        if (self.ball_falls == True) and (self.aver_z_vel > 0): # fall/nofall logic
            self.rebounds += 1
            self.ball_falls = False
            self.on_hit()
        elif (self.ball_falls == False) and (self.aver_z_vel < 0):
            self.ball_falls = True
            self.on_fall()

        if not self.ball_moving:
            pass

    def action(self):
        # X, Y, Z = self.getCords(self.ballId)
        # X_vel, Y_vel = self.getVel(self.ballId)[:2]
        alpha, beta = self.action_params[0]
        vel = self.action_params[1]
        delay = self.action_params[2]

        time_to_elevate = int(PLATFORM_ELEVATION / vel * fps)
        start_up_step = self.steps + delay
        start_down_step = self.steps + delay + time_to_elevate
        end_step = self.steps + delay + 2*time_to_elevate

        if self.debug:
            print('========== ACTION ', start_up_step, ' AND ', start_down_step)

        self.move(alpha, beta)
        self.timer.add_job(start_up_step, self.moveZ, vel, msg='UP', block_step=start_down_step)
        self.timer.add_job(start_down_step, self.moveZ, -vel, msg='DOWN', block_step=end_step)
        self.timer.add_job(end_step, self.move, 0, 0)


    def move(self, xx, yy):
        fo = 500 # force
        p.setJointMotorControl2(self.cubeId, 1, controlMode=p.POSITION_CONTROL, targetPosition = math.radians(xx), force = fo)
        p.setJointMotorControl2(self.cubeId, 2, controlMode=p.POSITION_CONTROL, targetPosition = math.radians(yy), force = fo)

    def moveZ(self, vel, msg, block_step = -1):
        p.setJointMotorControl2(self.cubeId, 0, controlMode=p.VELOCITY_CONTROL, targetVelocity=vel, force=500)
        if msg:
            if self.debug:
                print('============== step: {} sec: {}    {}'.format(self.steps, round(self.steps / fps, 3), msg))
            pass
        if block_step > 0: # block next exe untill block_step
            self.timer.block_exe(block_step)

    def make_simulation(self, ball_loc, ball_vel, action_params, debug):
        self.start(ball_loc=ball_loc, ball_vel=ball_vel, action_params=action_params, debug=debug)
        while True:
            if self.stop == True:
                reward = self.reward
                next_state = self.next_state_cords, self.next_state_vel
                self.purge()
                return reward, next_state, (self.A, self.B, self.C)
            env.step()
            time.sleep(self.fpss)

    def purge(self):
        p.removeBody(self.cubeId)
        p.removeBody(self.ballId)
######################################################################
    class Cheduler():
        def __init__(self):
            self.jobs = []
            self.block_step = -1

        def __call__(self, steps):
            try:
                if (self.jobs[0][0] <= steps) and (self.block_step < steps):
                    f = self.jobs.pop(0)
                    fun, args, kwargs = f[1]
                    fun(*args, **kwargs)
            except:
                pass

        def add_job(self, step, job, *args, **kwargs):
            if len(self.jobs) < 30: # in this configuration limit totally useless
                self.jobs.append([ step, [job, args, kwargs] ])
            else:
                if debug:
                    print('=== Too much jobs in Cheduler. Skip')
                    print(self.jobs)

        def block_exe(self, step): # block execution of jobs till N step
            self.block_step = step
            # print('Block set on ', step)

        # def clear(self):
        #     print('============= CLEARED')
        #     self.jobs.clear()

######################################################################
class Stats():
    def __init__(self):
        self.memory = []
        self.memory2 = []
        self.markers = []

    def safe_stat(self, cord):
        self.memory.append(cord)

    def safe_stat2(self, cord):
        self.memory2.append(cord)

    def show_stat(self):
        lin = np.linspace(0, secs, len(self.memory))
        plt.plot(self.memory, '-D', markevery=self.markers)
        plt.axis([0,stop_step,0,0.5])
        plt.show()

    def addMarker(self, mark):
        self.markers.append(mark)

normalize = lambda n, min, max: (n-min)/(max-min)
denormalize = lambda norm, min, max: norm*(max-min)+min

stats = Stats()
flag = False
env = Enviroment(fpss=0, GUI_enable=flag)
arr=[]

# general function for fast search
def my(x, *args):
    alpha, beta, vel, delay = x
    # print(alpha, beta, vel, delay, args)
    cords, ball_vel = args
    answ = env.make_simulation(cords, ball_vel, action_params=[(alpha, beta), vel, delay], debug=False)
    print(answ)
    return -answ[0]


ranges = ( slice(-15, 15, 5), slice(-15, 15, 5), slice(0.5, 1.5, 0.2), slice(10, 50, 10) )



# # # generate valid cases and safe them in csv
pairs = [(10, 10), (50, 10), (100, 5), (400, 10)]  # (sigma for velosity distribution, count of samples)
def generate_cases(pairs):
    gen_rand = lambda min, max: round(random.uniform(min, max), 3)
    arr = []

    # checks if the ball can touch the platform
    def check_reachability(cords, ball_vel):
        # 9999 is magic number for "check mode"
        res = env.make_simulation(cords, ball_vel, action_params=[(0, 0), 0.7, 9999], debug=False)
        if res[0] == 2:
            return True
        else:
            return False

    def gen_rand_vel(sigma, limit=600, loc=0):
        while True:
            num = np.random.normal(loc, sigma)
            if abs(num) <= limit:
                return num

    for pair in pairs:
        sigma, target_count = pair
        count = 0

        while count < target_count:
            x = gen_rand(-0.15, 0.15)
            y = gen_rand(-0.15, 0.15)
            z = gen_rand(0.08, 0.8)
            cords = (x, y, z)
            ball_vel = (gen_rand_vel(sigma), gen_rand_vel(sigma))

            if check_reachability(cords, ball_vel):
                count += 1
                arr.append([cords, ball_vel])
                print(arr[-1])

        print(len(arr))

    df = pd.DataFrame(arr, columns=["cords", "velocity"])
    df.to_csv('data/01_checked_dots.csv')

generate_cases(pairs)

# print(check_reachability( (0, 0, 0.3), (10, 10)  ) )

# res = brute(my, ranges, args=((0,0,0.3),(140,140)) )
# res = minimize(my, [9.965, -9.838,  0.512, 35.453], args=((0,0,0.3),(140,140)), options = {'maxiter': 10000})



# print( my( [9.07459748, -8.97687977,  0.577125,  38.14633508], (0,0,0.3),(140,140) )       )