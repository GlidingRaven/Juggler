import pybullet as p
import random, numpy, math, time
import pybullet_data
import matplotlib.pyplot as plt
import numpy as np
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
maxforce = 100
steps_to_trigger = 3
BALL_START_HEIGHT = 0.3
PLATFORM_ELEVATION = 0.2
WANTED_HEIGHT = 0.3

class Enviroment:
    def __init__(self):
        self.steps = 0
        self.rebounds = 0
        self.ball_moving = True # False if Z_velosity = 0
        self.ball_falls = True # False if Z_velosity > 0
        self.aver_z_vel = 0
        self.max_ball_height = BALL_START_HEIGHT
        self.contact_loc = (0, 0) # last contact location of ball-platform
        self.A = None
        self.B = None
        self.C = None

        p.connect(p.GUI)
        # p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        # p.setTimeStep(1. / 60)
        p.setPhysicsEngineParameter(enableSAT=1)
        self.cubeId = p.loadURDF("mycube.urdf", [0, 0, 0], globalScaling=1, useFixedBase=True, flags=p.URDF_INITIALIZE_SAT_FEATURES)
        self.makeBall()
        self.pushBall(self.ballId)
        p.resetDebugVisualizerCamera(1, 52, -32, (-0.26, 0.22, -0.24))

        self.slider = p.addUserDebugParameter("proportional", 0, 100, 80)
        self.der = p.addUserDebugParameter("derivative", 0, 100, 8)
        self.s1 = p.addUserDebugParameter("vel", 0, 3, 0.9)
        self.s2 = p.addUserDebugParameter("delay", 1, 150, 20)
        self.fps = p.addUserDebugParameter("fps", 1, 800, 240)
        self.timer = self.Cheduler()
        self.timer.add_job(self.steps + 3, self.predict_ball)

    def makeBall(self):
        self.ballId = p.loadURDF("myball.urdf", [0, 0, BALL_START_HEIGHT], globalScaling=1, flags=p.URDF_INITIALIZE_SAT_FEATURES)
        p.changeDynamics(self.ballId, -1, mass=1, restitution=0.8)

    def getCords(selfs, id):
        return np.around(p.getBasePositionAndOrientation(id)[0], 5)

    def getVel(selfs, id):
        return np.around(p.getBaseVelocity(id), 3)[0]

    def reaction(self, x, y, x_vel, y_vel, pro, der):
        if (self.steps % 1 == 0):
            self.move(y*pro+y_vel*der, -(x*pro+x_vel*der))

    # run when hit detected
    def on_hit(self):
        # stats.addMarker(self.steps)
        self.contact_loc = self.getCords(self.ballId)[:2]
        platZ = p.getLinkState(self.cubeId, 2)[0][2]
        print('===== HIT FUN', self.contact_loc, )
        x, y = self.contact_loc
        half = PLATFORM_ELEVATION / 2
        self.A = ((abs(x)*abs(y)*1/0.15) - (abs(x)+abs(y)) + 0.15)*(1/0.15)
        self.C = -abs(platZ-half)*(1/half)+1
        pass

    # run when ball begins fall
    def on_fall(self):
        print('===== FALL FUN')
        self.max_ball_height = self.getCords(self.ballId)[2]
        self.predict_ball()

    def reset(self):
        # self.steps = 0
        print('===== RESET')
        p.removeBody(self.cubeId)
        p.removeBody(self.ballId)
        self.cubeId = p.loadURDF("mycube.urdf", [0, 0, 0], globalScaling=1, useFixedBase=True, flags=p.URDF_INITIALIZE_SAT_FEATURES)
        self.makeBall()
        self.pushBall(self.ballId)
        self.timer.clear()
        self.timer.add_job(self.steps + 3, self.predict_ball)

    def pushBall(self, id):
        xf = random.uniform(-maxforce,maxforce)
        yf = random.uniform(-maxforce,maxforce)
        print('force applied: {} {}'.format(xf,yf))
        p.applyExternalForce(id, -1, [xf, yf, 0], [0,0,0], p.LINK_FRAME)

    def step(self):
        global fpss
        fpss = 1 / p.readUserDebugParameter(self.fps)

        self.steps += 1
        p.stepSimulation()
        self.timer(self.steps)

        X, Y, Z = self.getCords(self.ballId)
        ball_vel = self.getVel(self.ballId)

        self.triggers(ball_vel, (X, Y, Z))
        # platZ = p.getLinkState(0, 2, 1)[6][2]
        platZ = p.getLinkState(self.cubeId, 2)[0][2]
        print('{}    plat= {}  ballZ= {}  ballVel= {}  averVel= {}  isMoving= {}  isFall= {}'.format(
            self.steps , round(platZ, 3), round(Z, 3), ball_vel, self.aver_z_vel, str(self.ball_moving), str(self.ball_falls)))

        # self.reaction(X, Y, ball_vel[0], ball_vel[1], p.readUserDebugParameter(self.slider), p.readUserDebugParameter(self.der))
        self.reaction(X, Y, ball_vel[0], ball_vel[1], 80, 4)
        # pts = p.getContactPoints()
        # if len(pts) > 0:
        #     print("========== CONTACT", len(pts))

        if (Z < -0.1): # or (abs(X) > 0.2) or (abs(Y) > 0.2)
            self.reset()

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
            # print('s')
            pass

    def predict_ball(self):
        X, Y, Z = self.getCords(self.ballId)
        X_vel, Y_vel = self.getVel(self.ballId)[:2]

        B = -abs(Z - WANTED_HEIGHT) * (1 / WANTED_HEIGHT) + 1
        self.B = 0 if B < 0 else B
        # print('B = ', self.B , Z)

        # print(X, Y, Z, X_vel, Y_vel)
        z1 = p.readUserDebugParameter(self.s1)
        z2 = p.readUserDebugParameter(self.s2)
        mul = p.readUserDebugParameter(self.slider)
        # z1 = 1
        # z2 = 20
        # mul = 80
        # self.move(Y * mul, -X * mul)
        vel = round(z1, 3)
        delay = int(z2)
        time_to_elevate = int(PLATFORM_ELEVATION / vel * fps)
        # print(PLATFORM_ELEVATION, vel, fps, time_to_elevate)
        start_up_step = self.steps + delay
        start_down_step = self.steps + delay + time_to_elevate
        end_step = self.steps + delay + 2*time_to_elevate

        print('========== PREDICT ', start_up_step, ' AND ', start_down_step)

        self.timer.add_job(start_up_step, self.moveZ, vel, msg='UP', block_step=start_down_step)
        self.timer.add_job(start_down_step, self.moveZ, -vel, msg='DOWN', block_step=end_step)


    def move(self, xx, yy):
        fo = 500 # force
        # print(p.getEulerFromQuaternion(p.getLinkState(self.cubeId, 2)[1])[1])
        p.setJointMotorControl2(self.cubeId, 1, controlMode=p.POSITION_CONTROL, targetPosition = math.radians(xx), force = fo)
        p.setJointMotorControl2(self.cubeId, 2, controlMode=p.POSITION_CONTROL, targetPosition = math.radians(yy), force = fo)

    def moveZ(self, vel, msg, block_step = -1):
        # print('======= MOVEz called')
        p.setJointMotorControl2(self.cubeId, 0, controlMode=p.VELOCITY_CONTROL, targetVelocity=vel, force=500)
        if msg:
            print('============== step: {} sec: {}    {}'.format(self.steps, round(self.steps / fps, 3), msg))
        if block_step > 0:
            self.timer.block_exe(block_step)

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
            if len(self.jobs) < 2:
                self.jobs.append([ step, [job, args, kwargs] ])
            else:
                print('=== Too much jobs in Cheduler. Skip')
                print(self.jobs)

        def block_exe(self, step): # block execution of jobs till N step
            self.block_step = step
            print('Block set on ', step)

        def clear(self):
            print('============= CLEARED')
            pass
            self.jobs.clear()


# p.changeDynamics(cubeId, 0, restitution=1)
# print('ball: ', p.getDynamicsInfo(ballId, -1) )
# print('cube: ', p.getDynamicsInfo(cubeId, -1) )

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


stats = Stats()
env = Enviroment()
# print('cam:', p.getDebugVisualizerCamera())
while True:
  env.step()
  time.sleep(fpss)
  # time.sleep(0.0041) # 240fps