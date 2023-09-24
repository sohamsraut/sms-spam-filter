from sklearn.naive_bayes import *
from sklearn.dummy import *
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
import pandas

def perform(classifiers, vecs, train, test):
    for classifier in classifiers:
        for vec in vecs:
            string = ''
            string += classifier.__class__.__name__ + ' with ' + vec.__class__.__name__

            vec_text = vec.fit_transform(train.v2)
            classifier.fit(vec_text, train.v1)

            vec_text = vec.transform(test.v2)
            score = classifier.score(vec_text, test.v1)
            string += ' score = ' + str(score)
            print(string)

data = pandas.read_csv("spam.csv", encoding='latin-1')
learn = data[:4400]
test = data[4400:]

classifiers = [
    BernoulliNB(),
    RandomForestClassifier(n_estimators=100, n_jobs=-1),
    AdaBoostClassifier(),
    KNeighborsClassifier(),
    OneVsRestClassifier(LogisticRegression()),
    OneVsRestClassifier(SVC(kernel='linear')),
    DummyClassifier(),
    SGDClassifier(),
    RidgeClassifier(),
    RidgeClassifierCV(),
    ExtraTreesClassifier(),
    GradientBoostingClassifier(),
    DecisionTreeClassifier(),
    CalibratedClassifierCV(),
    PassiveAggressiveClassifier()
  ]
vecs = [CountVectorizer(), TfidfVectorizer(), HashingVectorizer()]
perform(classifiers, vecs, learn, test)
