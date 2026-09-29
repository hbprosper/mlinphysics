import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
# -----------------------------------------------------------
class ExactDensity:

    def __init__(self):
        # Class y = 1
        self.rv1 = st.multivariate_normal(mean=[0.25, -0.25], 
                                          cov=[[0.20, 0.24], 
                                               [0.24, 0.40]])
        # Class y = 0
        self.rv2 = st.multivariate_normal(mean=[-0.10, 0.10], 
                                          cov=[[0.60, 0.40], 
                                               [0.40, 0.30]])

    def generate(self, filename=None, N=50_000):
        d1 = self.rv1.rvs(N)
        d2 = self.rv2.rvs(N)

        ones = np.ones(N)[:, np.newaxis]
        dd1  = np.hstack([d1, ones])  # stack horizontally

        zeros= np.zeros(N)[:, np.newaxis]
        dd2  = np.hstack([d2, zeros])

        dd = np.vstack([dd1, dd2])    # stack vertically
        np.random.shuffle(dd)

        df = pd.DataFrame(dd, columns=['x1', 'x2', 'y'])

        # save dataframe if requested
        if filename != None:
            df.to_csv(filename, index=False)

        return df

    def grid(self, flatten=False, xmin=-1, xmax=1, ymin=-1, ymax=1, N=100):
        xstep = (xmax-xmin)/N
        ystep = (ymax-ymin)/N
        x1, x2 = np.mgrid[xmin:xmax+xstep/2:xstep, ymin:ymax+ystep/2:ystep]
        if flatten:
            x1 = x1.flatten()
            x2 = x2.flatten()
        return x1, x2

    def __call__(self, x1, x2):
        # compute density at a grid of (x1, x2) points
        pos= np.dstack((x1, x2))
        p1 = self.rv1.pdf(pos)
        p0 = self.rv2.pdf(pos)
        return (p1 + p0)/2

    def prob(self, x1, x2):
        # compute p(y=1|x) at a grid of (x1, x2) points
        pos= np.dstack((x1, x2))
        p1 = self.rv1.pdf(pos)
        p0 = self.rv2.pdf(pos)
        return p1 / (p1 + p0)

    def plot(self, papprox=None, dataframe=None,
            xmin=-1, xmax=1, ymin=-1, ymax=1, fgsize=(5, 5)):
        x1, x2 = self.grid()

        fig = plt.figure(figsize=fgsize)
        ax  = fig.add_subplot(111)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        tickmarks = [-1.0, -0.5, 0.0, 0.5, 1.0]
        ax.set_xticks(tickmarks)
        ax.set_yticks(tickmarks)

        # plot p(x)
        plot_approx = type(papprox) != type(None)
        if not plot_approx:
            ax.contourf(x1, x2, self(x1, x2))

        if type(dataframe) != type(None):
            df = dataframe
            K = 1000
            xp, yp, _ = df[df.y > 0.5].to_numpy().T
            ax.scatter(xp[:K], yp[:K], s=0.8, color='cyan')

            xp, yp, _ = df[df.y < 0.5].to_numpy().T
            ax.scatter(xp[:K], yp[:K], s=0.8, color='red')

        # plot p(1|x)
        levels = np.arange(0.1, 1.0, 0.2) # values of p(1|x)
        pexact = self.prob(x1, x2)
        ax.contour(x1, x2, pexact, levels=levels, cmap='rainbow_r', linestyles='solid')

        if plot_approx:
            ax.contour(x1, x2, papprox, levels=levels, cmap='rainbow_r', linestyles='dashed')

        fig.tight_layout()
        plt.show()
