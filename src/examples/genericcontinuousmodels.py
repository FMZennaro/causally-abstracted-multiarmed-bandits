
import numpy as np
import scipy.stats as stats
import pandas as pd


class XYZchain:
    def __init__(self, px=stats.norm().rvs, fx=lambda ux: ux,
                 py=stats.norm().rvs, fy=lambda uy, x: x,
                 pz=stats.norm().rvs, fz=lambda uz, y: y,
                 labels=['U_X', 'U_Y', 'U_Z', 'X', 'Y', 'Z']):
        self.px = px
        self.py = py
        self.pz = pz

        self.fx = fx
        self.fy = fy
        self.fz = fz

        self.labels = labels

    def sample_X(self):
        ux = self.px()
        x = self.fx(ux)
        return ux, x

    def sample_Y(self, x):
        uy = self.py()
        y = self.fy(uy, x)
        return uy, y

    def sample_Z(self, y):
        uz = self.pz()
        z = self.fz(uz, y)
        return uz, z

    def sample(self):
        ux, x = self.sample_X()
        uy, y = self.sample_Y(x)
        uz, z = self.sample_Z(y)
        return ux, uy, uz, x, y, z

    def generate_obs_dataset(self, n_samples=1000):
        D = np.array([self.sample() for _ in range(n_samples)]).reshape((n_samples, len(self.labels)))
        return pd.DataFrame(D, columns=self.labels)

    def generate_intX_dataset(self, intX, n_samples=1000):
        natural_fx = self.fx
        self.fx = lambda ux: intX
        df = self.generate_obs_dataset(n_samples=n_samples)
        self.fx = natural_fx
        return df

    def generate_intY_dataset(self, intY, n_samples=1000):
        natural_fy = self.fy
        self.fy = lambda uy, x: intY
        df = self.generate_obs_dataset(n_samples=n_samples)
        self.fy = natural_fy
        return df

    def generate_intXY_dataset(self, intX, intY, n_samples=1000):
        natural_fx = self.fx;
        natural_fy = self.fy
        self.fx = lambda ux: intX;
        self.fy = lambda uy, x: intY
        df = self.generate_obs_dataset(n_samples=n_samples)
        self.fx = natural_fx;
        self.fy = natural_fy
        return df


class XYchain:
    def __init__(self, px=stats.norm().rvs, fx=lambda ux: ux,
                 py=stats.norm().rvs, fy=lambda uy, x: x,
                 labels=['U_X', 'U_Y', 'X', 'Y']):
        self.px = px
        self.py = py

        self.fx = fx
        self.fy = fy

        self.labels = labels

    def sample_X(self):
        ux = self.px()
        x = self.fx(ux)
        return ux, x

    def sample_Y(self, x):
        uy = self.py()
        y = self.fy(uy, x)
        return uy, y

    def sample(self):
        ux, x = self.sample_X()
        uy, y = self.sample_Y(x)
        return ux, uy, x, y

    def generate_obs_dataset(self, n_samples=1000):
        D = np.array([self.sample() for _ in range(n_samples)]).reshape((n_samples, len(self.labels)))
        return pd.DataFrame(D, columns=self.labels)

    def generate_intX_dataset(self, intX, n_samples=1000):
        natural_fx = self.fx
        self.fx = lambda ux: intX
        df = self.generate_obs_dataset(n_samples=n_samples)
        self.fx = natural_fx
        return df