from scipy import stats
from sklearn.preprocessing import RobustScaler
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ks_shapiro_boolean(vector):
    '''
    return True if vector is gaussian-like with kolmogorov/smirnov and shapiro/wilk tests.
    '''
       
    critical_threshold = 0.001
    shapiro_ok = False
    ks_ok = False
    
    # standardization
    X = vector.values.reshape(-1,1)
    standardized_X = RobustScaler().fit_transform(X)
    
    # shapiro test
    small_size_standardized_X = np.random.choice(standardized_X.reshape(-1), size=10, replace=False)
    if stats.shapiro(small_size_standardized_X)[1] >= critical_threshold:
        shapiro_ok = True
            
    # kolmogorov/smirnov test
    random_normal = np.random.normal(loc=standardized_X.mean(), 
                                     scale=standardized_X.std(), 
                                     size=(standardized_X.shape[0], 1))

    if stats.kstest(standardized_X.reshape(-1), random_normal.reshape(-1), 
                    N=standardized_X.shape[0])[1] >= critical_threshold :
        ks_ok = True

    if (shapiro_ok==True) & (ks_ok==True): 
        return True


def elbow_method_graph(matrix, max_n_clusters):
    
    from sklearn.cluster import KMeans
    
    distance = []

    # sum of distance when n_cluster=k
    k_range = range(1,max_n_clusters)
    
    for k in range(1,max_n_clusters):
        model = KMeans(n_clusters=k, random_state=0).fit(matrix)
        distance.append(model.inertia_)
    
    # visual result
    plt.figure(dpi=100)
    plt.plot(k_range, distance)
    plt.xlabel('Clusters number')
    plt.ylabel('Sum of distances')
    

def confusion_matrix_sens_spec(model, y, X):
    import sklearn.metrics as metrics
    
    # model prediction
    prediction = model.predict_proba(X)[:,1]
    prediction_boolean = model.predict(X)

    # model scoring
    confusion_matrix = metrics.confusion_matrix(y_true=y, y_pred=prediction_boolean)                       
    print(confusion_matrix)

    print('sensitivity +', (confusion_matrix[1, 1] / 
                           (confusion_matrix[1, 1] + confusion_matrix[0, 1])))

    print('specificity -', (confusion_matrix[0, 0] / 
                           (confusion_matrix[0, 0] + confusion_matrix[1, 0])))
    
    print('AUROC score :',metrics.roc_auc_score(y_true=y, y_score=model.predict(X)))
    

def logistic_regression(y, X):
    
    # model selection
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV

    # X = select Feature, y = Target
    X = np.array(X)
    y = np.array(y).reshape(-1,1)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size=0.2,
                                                        random_state=0)
 
    # GridSearchCV
    parameters_dictionnay = {'penalty': ['l1', 'l2', 'none'],
                             'solver' : ['lbfgs', 'liblinear'],
                             'class_weight' : ['none', 'balanced'],
                             'fit_intercept' : ['True', 'False']
                             }

    grid = GridSearchCV(LogisticRegression(), parameters_dictionnay, cv=10)
    grid.fit(X_train, y_train)
    
    # scoring then modelling
    print('GridSearchCV : ', grid.best_params_)
    model = grid.best_estimator_

    #model.fit(X_train,y_train)
    print('Coefficients : ', model.coef_)
    
    # model score
    print('Train score (CV 20):', cross_val_score(LogisticRegression(),
                                               X_train, y_train,
                                               cv=20).mean())

    print('Test score :', model.score(X_test, y_test))
    
    return model

def bernoulli_ks_pvalue(column):
    from scipy import stats

    # equiprobability sampling
    notes_true_equal_false_size = np.random.choice(column[column==True], 
                                             size=column[column==False].shape[0], 
                                             replace=False)

    notes_equiprobability = np.concatenate([notes_true_equal_false_size,
                                           column[column==False]], 
                                           axis=0)

    # random binomial variable
    random_binomial = np.random.binomial(n=1, p=0.5, size=notes_equiprobability.shape[0])

    return stats.kstest(notes_equiprobability, 
                        random_binomial, 
                        N=notes_equiprobability.shape[0])[1]

def pca_transformation(matrix, n_components):
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler
    
    scaled_matrix = RobustScaler(with_centering=True, with_scaling=False).fit_transform(matrix)

    pca = PCA(n_components=n_components)
    pca.fit(scaled_matrix)

    components = pd.DataFrame(pca.components_, 
                              columns=matrix.columns, 
                              index=np.arange(1, pca.components_.shape[0]+1))
    
    return components, pca


def pca_components(components, matrix, dataframe):
    
    from sklearn.preprocessing import RobustScaler
    
    scaled_matrix = RobustScaler(with_centering=True, with_scaling=False).fit_transform(matrix)
   
    F1 = (  scaled_matrix[:, 0] * components.iloc[0, 0] 
          + scaled_matrix[:, 1] * components.iloc[0, 1]
          + scaled_matrix[:, 2] * components.iloc[0, 2])

    F2 = (  scaled_matrix[:, 0] * components.iloc[1, 0] 
          + scaled_matrix[:, 1] * components.iloc[1, 1]
          + scaled_matrix[:, 2] * components.iloc[1, 2])

    dataframe['F1'] = F1
    dataframe['F2'] = F2
    
    
def testing_dataframe(dataframe, model):
    
    # probability column
    dataframe['prediction'] = (model.predict_proba(dataframe[['F1','F2']])[:,0]) * 100
    dataframe['prediction'] = dataframe['prediction'].round(2)

    # is_genuine column
    dataframe['is_genuine'] = np.where(dataframe['prediction'] >= 50, 1, 0)
    dataframe['is_genuine'] = dataframe['is_genuine'].astype('bool')

    # third step : visual result
    return dataframe[['id', 'prediction', 'is_genuine']].set_index('id')


def pca_scree(pca, savefig):
    
    # scree
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.xlabel("Composantes")
    plt.ylabel("% de Variance")

    # kaiser mean criterion
    kaiser_criterion = pca.explained_variance_ratio_.mean()
    plt.axhline(y=(pca.explained_variance_ratio_.mean()*100), color='r')
    
    # dumping graph
    if savefig:
        plt.tight_layout()
        plt.savefig('P6_03_screeplot.png')

    plt.show()
    
    
def pca_correlation_circle(pca, component_plane, columns, savefig):
    
    # component_plane transformation
    component_plane = np.array(component_plane)
    component_plane -= 1
    
    # initialisation de la figure
    fig, ax = plt.subplots(figsize=(7,7))
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    
    # affichage du cercle
    circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='black')
    plt.gca().add_artist(circle)
   
    # affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # affichage des flèches
    plt.quiver(np.zeros(pca.components_.shape[1]), 
               np.zeros(pca.components_.shape[1]),
               pca.components_[component_plane[0],:], pca.components_[component_plane[1],:], 
               angles='xy', scale_units='xy', scale=1, color="grey")
            
    # affichage des noms des variables   
    for i,(x, y) in enumerate(pca.components_[[component_plane[0],component_plane[1]]].T):
        plt.text(x, y, columns.columns[i], 
                 fontsize='15', ha='center', va='bottom', color="red")

    # nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(component_plane[0]+1, int(round(100*pca.explained_variance_ratio_[component_plane[0]],0))))
    plt.ylabel('F{} ({}%)'.format(component_plane[1]+1, int(round(100*pca.explained_variance_ratio_[component_plane[1]],0))))
    
    # dumping graph
    if savefig:
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig('P6_04_correlation_circle_'+ str(component_plane[0] + 1) + '_' 
                    + str(component_plane[1] + 1) + '.png')
    
    plt.show()
    
    
def pca_factorial_plane(pca, component_plane, groups, columns, labels, centroids, savefig):
    
    import matplotlib.colors as colors
    
    # component_plane transformation
    component_plane = np.array(component_plane)
    component_plane -= 1
    
    # matrix preparation
    matrix = pca.fit_transform(columns)
    
    # figure preparation
    fig = plt.figure(figsize=(7,7))
    max_dimension = 3
    plt.xlim([-max_dimension, max_dimension])
    plt.ylim([-max_dimension, max_dimension])
    
    # components axes
    plt.plot([-100, 100], [0, 0], color='grey', ls='--', alpha=0.3)
    plt.plot([0, 0], [-100, 100], color='grey', ls='--', alpha=0.3)
    
    cmap = colors.LinearSegmentedColormap.from_list('', ['lightcoral','mediumseagreen'])
    plt.scatter(matrix[:, component_plane[0]], matrix[:, component_plane[1]], c=groups, cmap=cmap)  
    
    # centroids visual
    if centroids:
        
        # centroids computation
        centroids = np.concatenate([matrix,np.array(groups, dtype=int).reshape(-1, 1)], axis=1)
        centroids = pd.DataFrame(centroids).groupby(3).median()
        
        # red dots
        plt.scatter(centroids.iloc[:, component_plane[0]], centroids.iloc[:, component_plane[1]], 
                    c='black', alpha=0.7)
        
        # text
        [plt.text(x=centroids.iloc[i, component_plane[0]], 
                  y=centroids.iloc[i, component_plane[1]], 
                  s=int(centroids.index[i]), 
                  c='black',
                  fontsize=20) for i in np.arange(0, centroids.shape[0])];
                    
    # naming dots
    if labels:
        [plt.text(x=matrix[i, component_plane[0]], 
                  y=matrix[i, component_plane[1]], 
                  s=columns.index[i], 
                  c='r',
                  fontsize=8, alpha=0.7) for i in np.arange(0, columns.shape[0])];

    # nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(component_plane[0]+1, int(round(100*pca.explained_variance_ratio_[component_plane[0]],0))))
    plt.ylabel('F{} ({}%)'.format(component_plane[1]+1, int(round(100*pca.explained_variance_ratio_[component_plane[1]],0))))
    
    # dumping graph
    if savefig:
        plt.tight_layout()
        plt.savefig('P6_05_factorial_plane_'+ str(component_plane[0] + 1) + '_' 
                    + str(component_plane[1] + 1) + '.png')
    plt.show()