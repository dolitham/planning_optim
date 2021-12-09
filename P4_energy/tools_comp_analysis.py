from numpy import cumsum
from prince import PCA, MCA
import matplotlib.pyplot as plt


def _display_results(inertia, cumulative_inertia, type):
    plt.figure()
    plt.plot(inertia)
    plt.title('explained inertia per ' + type + ' component')
    plt.show()

    plt.figure()
    plt.plot(cumulative_inertia)
    plt.title('Cumulative inertia per ' + type + ' component')
    plt.show()


def _get_ac_transform(x_train, x_test, cut_rate, type, display):
    if ''.join(sorted(type.lower())) == 'acp':
        ac = PCA(n_components=x_train.shape[1])
    else:
        ac = MCA(n_components=500)
    ac.fit(x_train)
    inertia = ac.explained_inertia_
    cumulative_inertia = cumsum(inertia)
    nb_comp_to_keep = list(cumulative_inertia >= cut_rate).index(True) if cut_rate < 1 else len(
        cumulative_inertia)

    def transform(x):
        return ac.transform(x).iloc[:, :nb_comp_to_keep]

    if display:
        _display_results(inertia, cumulative_inertia, type)

    return transform(x_train).add_prefix(type + '_'), transform(x_test).add_prefix(type + '_')


def get_acp(x_train, x_test, cut_rate=0.8, display=False):
    return _get_ac_transform(x_train, x_test, cut_rate, 'PCA', display=display)


def get_acm(x_train, x_test, cut_rate=0.8, display=False):
    return _get_ac_transform(x_train, x_test, cut_rate, 'MCA', display=display)
