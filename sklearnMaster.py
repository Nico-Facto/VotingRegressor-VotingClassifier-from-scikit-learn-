"""
From sickit Learn 0.22 -- Built on 01/2020

module includes classes and functions to call and visualize performance of 3 models
and the fusion call VotingRegressor or VotingClassifier. it gives the possibility 
of adjusting the vote with weights according to the performances of the various models

This is to simplify access to the functions created by sickit learn see :

https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingRegressor.html
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html

"""
# Author: Nicolas Autexier <nicolas.atx@gmx.fr>,

import matplotlib.pyplot as plt
import numpy as np

class voter():
    """
    return Voting & scores (on val test) for your models
    3 models with weight set to 1 by default, pass an array with your weights(optional)
    from sklearn, see VotingRegressor or ClassifierRegressor

    Parameters
    ----------
    values : 

    mod = classifier or regressor as str with quotes
    Pass your split =  x_train, x_val, y_train, y_val
    weights=[1, 1, 1] give weight for models impact on Voter

    Returns
    -------
     .get_model() return only the voting model

    """

    def __init__(self, mod, x_train, x_val, y_train, y_val,  weights=[1, 1, 1]):

        self.mod = mod
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.weights = weights

    def get_model(self):
        ''' return model create by voter'''

        if self.mod == "regressor" :

            from sklearn.tree import DecisionTreeRegressor
            from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
            from sklearn.metrics import mean_squared_error as MSE
            import sklearn.metrics as SM

            self.mod1 = GradientBoostingRegressor(criterion='mae', n_estimators=200, max_depth=5)
            self.mod2 = RandomForestRegressor(criterion='mae', n_estimators=200, max_depth=5)
            self.mod3 = DecisionTreeRegressor(criterion='mae', splitter='best',max_depth=5)

            self.vtr = VotingRegressor(estimators=[('gb', self.mod1), ('rf', self.mod2), ('lr', self.mod3)], weights=self.weights)

            self.mod1 = self.mod1.fit(self.x_train, self.y_train)
            self.mod2 = self.mod2.fit(self.x_train, self.y_train)
            self.mod3 = self.mod3.fit(self.x_train, self.y_train)
            self.vtr = self.vtr.fit(self.x_train, self.y_train)

            xt = self.x_train[:50]

            plt.figure(figsize=(20,10))
            plt.plot(self.mod1.predict(xt), 'gd', label='GradientBoostingRegressor')
            plt.plot(self.mod2.predict(xt), 'b^', label='RandomForestRegressor')
            plt.plot(self.mod3.predict(xt), 'ys', label='DecisionTreeRegressor')
            plt.plot(self.vtr.predict(xt), 'r*', label='VotingRegressor')

            plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            plt.ylabel('predicted')
            plt.xlabel('training samples')
            plt.legend(loc="best")
            plt.title('Comparison of individual predictions with averaged')
            plt.show()

            print("Model Voting")
            vote_pred = self.vtr.predict(self.x_val)
            RMSE = np.sqrt(MSE(vote_pred, self.y_val))
            score = SM.mean_absolute_error(vote_pred,self.y_val)
            
            print("RMSE on val = ",RMSE.round(4))
            print("MAPE on val = ", score)
            print("")

            print("Model GradientBoostingRegressor")
            
            mod1_pred = self.mod1.predict(self.x_val)

            RMSE = np.sqrt(MSE(mod1_pred, self.y_val))
            score = SM.mean_absolute_error(mod1_pred, self.y_val)
            
            print("RMSE on val = ",RMSE.round(4))
            print("MAPE on val = ", score)
            print("")

            print("Model RandomForestRegressor")
            
            mod2_pred = self.mod2.predict(self.x_val)

            RMSE = np.sqrt(MSE(mod2_pred, self.y_val))
            score = SM.mean_absolute_error(mod2_pred, self.y_val)
            
            print("RMSE on val = ",RMSE.round(4))
            print("MAPE on val = ", score)
            print("")

            print("Model DecisionTreeRegressor")
            
            mod3_pred = self.mod3.predict(self.x_val)

            RMSE = np.sqrt(MSE(mod3_pred, self.y_val))
            score = SM.mean_absolute_error(mod3_pred, self.y_val)
            
            print("RMSE on val = ",RMSE.round(4))
            print("MAPE on val = ", score)
            print("")

            return self.vtr

        elif self.mod == "classifier" :

            from sklearn.linear_model import LogisticRegression
            from sklearn.naive_bayes import GaussianNB
            from sklearn.ensemble import RandomForestClassifier, VotingClassifier
            import sklearn.metrics as SM

            self.clf1 = LogisticRegression(max_iter=3000, random_state=42, solver='lbfgs')
            self.clf2 = RandomForestClassifier(n_estimators=100, random_state=123)
            self.clf3 = GaussianNB()

            self.vtc = VotingClassifier(estimators=[('lr', self.clf1), ('rf', self.clf2), ('gnb', self.clf3)], voting='soft', weights=self.weights)

            # predict class probabilities for all classifiers
            probas = [c.fit(self.x_train, self.y_train).predict_proba(self.x_train) for c in (self.clf1, self.clf2, self.clf3, self.vtc)]

            # get class probabilities for the first sample in the dataset
            class1_1 = [pr[0, 0] for pr in probas]
            class2_1 = [pr[0, 1] for pr in probas]


            # plotting

            N = 4  # number of groups
            ind = np.arange(N)  # group positions
            width = 0.35  # bar width

            fig, ax = plt.subplots(figsize=(20,10))

            # bars for classifier 1-3
            p1 = ax.bar(ind, np.hstack(([class1_1[:-1], [0]])), width,
                        color='green', edgecolor='k')
            p2 = ax.bar(ind + width, np.hstack(([class2_1[:-1], [0]])), width,
                        color='lightgreen', edgecolor='k')

            # bars for VotingClassifier
            p3 = ax.bar(ind, [0, 0, 0, class1_1[-1]], width,
                        color='blue', edgecolor='k')
            p4 = ax.bar(ind + width, [0, 0, 0, class2_1[-1]], width,
                        color='steelblue', edgecolor='k')

            # plot annotations
            plt.axvline(2.8, color='k', linestyle='dashed')
            ax.set_xticks(ind + width)
            ax.set_xticklabels([f'LogisticRegression\nweight {self.weights[0]}',
                                f'GaussianNB\nweight {self.weights[1]}',
                                f'RandomForestClassifier\nweight {self.weights[2]}',
                                'VotingClassifier\n(average probabilities)'],
                            rotation=40,
                            ha='right')

            plt.ylim([0, 1])
            plt.title('Class probabilities for sample 1 by different classifiers')
            plt.legend([p1[0], p2[0]], ['class 1', 'class 2'], loc='upper left')
            plt.tight_layout()
            plt.show()

            print("Model VotingClassifier")
            vote_pred = self.vtc.predict(self.x_val)
            score = SM.accuracy_score(vote_pred,self.y_val)
            
            print("Accuracy = ", score.round(4))
            print("")

            print("Model LogisticRegression")
            
            vote_pred = self.clf1.predict(self.x_val)
            score = SM.accuracy_score(vote_pred,self.y_val)
            
            print("Accuracy = ", score.round(4))
            print("")
        
            print("Model GaussianNB")
            
            vote_pred = self.clf3.predict(self.x_val)
            score = SM.accuracy_score(vote_pred,self.y_val)
            
            print("Accuracy = ", score.round(4))
            print("")

            print("Model RandomForestClassifier")
            
            vote_pred = self.clf2.predict(self.x_val)
            score = SM.accuracy_score(vote_pred,self.y_val)
            
            print("Accuracy = ", score.round(4))
            print("")
            return self.vtc

       








