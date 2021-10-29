from Juggler import Juggler, Predictors

stats = Juggler.Stats()
env = Juggler.Enviroment_continuous(fpss=0, GUI_enable=True)
oracle = Predictors.ActionPredictor('action_model.pickle')
# oracle.fit('03_for_train.csv')
# oracle.save('action_model.sav')

# Run 200 games in GUI mode
for _ in range(200):
    env.game(oracle)