import matplotlib.pyplot as plt
import numpy as np


data = np.loadtxt("groupA.txt", delimiter=",")
x = data[:, :2]
y= data[:,2].astype(int)

rng = np.random.default_rng(42)
idx = rng.permutation(len(x))
cut = int(.75 * len(x))
train_idx, test_idx = idx[:cut], idx[cut:]

x_train, y_train = x[train_idx], y[train_idx]
x_test, y_test = x[test_idx], y[test_idx]

np.savetxt("train.txt", np.column_stack((x_train, y_train)), fmt="%.6f", delimiter=",")
np.savetxt("test.txt", np.column_stack((x_test, y_test)), fmt="%.6f", delimiter=",")



def plot_decision_planes(weights_history, patterns, labels, step=1):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')

    for p, d in zip(patterns, labels):
        color = 'r' if d == -1 else 'b'
        ax.scatter(p[0], p[1], p[2], c=color, s=80, edgecolor='k')

    x_range = np.linspace(0, 3, 20)
    y_range = np.linspace(0, 3, 20)
    X, Y = np.meshgrid(x_range, y_range)

    idxs = list(range(0, len(weights_history), step))
    cmap = plt.cm.get_cmap('RdYlGn')
    n = len(idxs)

    for j, i in enumerate(idxs):
        w = weights_history[i]
        if w[2] == 0:
            continue  # skip vert planes
        Z = -(w[0]*X + w[1]*Y) / w[2]
        color = cmap(j / max(n-1, 1))
        ax.plot_surface(X, Y, Z, alpha=0.15, linewidth=0, color=color, antialiased=False)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3 (bias)")
    ax.set_title("Perceptron Decision Boundary Evolution")
    ax.set_box_aspect((1,1,1))
    norm = plt.Normalize(vmin=0, vmax=n-1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.6)
    cb.set_label("old → new")
    cb.set_ticks([0, n-1]); cb.set_ticklabels(["oldest", "newest"])

    plt.show()


def sign(net):
    if net >= 0:
        return 1
    else:
        return -1
    
def printdata(iteration, pattern, net, err, learn, ww):
    #ite= 0, p=0, net= 4.0, err= -2.000, lrn= -.200, w=[1.00, 3.00, -3.00]
    ww_formatted = ['%.2f' % elem for elem in ww]
    print(f"ite={iteration:2d}, p={pattern:2d}, net={net:5.1f}, err={err:7.3f}, lrn={learn:7.3f}, w={ww_formatted}")

ite=30
npat=2
ni=3
alpha=0.1
ww=[1.0, 3.0, -3.0]
pat=([1,2,1], [2,1,1])
dout= (-1,1)

weights_history = [ww.copy()]
errors_history = []


#print(type(ww))
for iteration in range(0, ite): 
    ou = [0,0] 
    for pattern in range(0,npat): 
        net=0
        for i in range(0,ni):
            net = net + ww[i]*pat[pattern][i]
            
        ou[pattern] = sign(net)
        err = dout[pattern] - ou[pattern]
        learn = alpha*err
        printdata(iteration,pattern,net,err,learn,ww)
        for i in range(0,ni):
            ww[i] = ww[i] + learn*pat[pattern][i]
        weights_history.append(ww.copy())

plot_decision_planes(weights_history, pat, dout, step=5)