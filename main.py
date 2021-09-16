import Juggler
import Predictors


stats = Juggler.Stats()
flag = True
env = Juggler.Enviroment(fpss=0, GUI_enable=flag)
oracle = Predictors.Predictor('03_for_train.csv')
for _ in range(200):
    env.game(oracle)