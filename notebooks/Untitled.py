# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# import libraries
import numpy as np
import matplotlib.pyplot as plt
from pybqreg import BayesianQuantileRegression

# %%
import scipy.stats as sts
import arviz as av
from arviz import hdi as av_hdi

# %%
N = 2000
hN = N // 2

# %%
# rng = np.random.Generator(np.random.PCG64DXSM(23))
rng = np.random.Generator(np.random.PCG64DXSM())


# %%
# M = 1000
# mu_trend = np.linspace(1, 2, num=N)
# raw_sample = np.exp(sts.norm(loc=rng.normal(scale=1., size=N) + mu_trend).rvs((M, N), rng))
# mean_sample = raw_sample.mean(0)
# sample = mean_sample + rng.normal(size=N)

# %%
def _draw_pdf(dist, n=1000, limit=1e-6):
    x = np.linspace(dist.ppf(limit), dist.ppf(1. - limit), num=n)
    return x, dist.pdf(x)


# %%
b0_dist = sts.lognorm(1)
b1_dist = sts.norm(scale=0.1)

betas = np.empty((2,1))
betas[0] = b0_dist.rvs(random_state=rng)
betas[1] = b1_dist.rvs(random_state=rng)

fig, axs = plt.subplots(figsize=(12, 4), ncols=2)
_ = axs[0].plot(*_draw_pdf(b0_dist))
axs[0].axvline(betas[0], ls="--", c="red", label=np.round(betas[0], 4)[0])
_ = axs[1].plot(*_draw_pdf(b1_dist))
axs[1].axvline(betas[1], ls="--", c="red", label=np.round(betas[1], 4)[0])
axs[0].legend()
axs[0].set_title(r"$\beta_0\sim LN(1)$")
axs[1].set_title(r"$\beta_1\sim N(0, 0.1)$")
axs[1].legend()

# %%
X = np.ones((N, 2))
X[:, 1] = np.linspace(0, 1, num=N)
y = np.matmul(X, betas).ravel() + rng.normal(scale=0.001, size=N)

# %%
fig, ax = plt.subplots(figsize=(24, 8))
ax.plot(y)

# %%
y_t = y[:hN]
X_t = X[:hN, :]

# %%
# set the target quantile (tau) and run the Gibbs sampler
p = 0.50
theta = (1 - 2 * p) / (p * (1 - p))
tau = np.sqrt(2 / (p * (1 - p)))

# %%
p_v = np.linspace(1e-4, 1-1e-4, 200)
theta_v = (1 - 2 * p_v) / (p_v * (1 - p_v))
tau_v = np.sqrt(2 / (p_v * (1 - p_v)))

# %%
plt.plot(p_v, theta_v)
plt.plot(p_v, tau_v)
plt.yscale("log")

# %%
# initialize object
obj = BayesianQuantileRegression(y_t, X_t)
obj.set_omp_n_threads(4)

# %%
# set prior pars
beta0 = np.zeros(2)
beta_hat = np.linalg.solve( np.matmul(X_t.T, X_t), np.matmul(X_t.T, y_t) )


V0 = np.eye(2) * 1000

prior_shape = 3
prior_scale = 3
obj.set_prior_params(beta0, V0, prior_shape, prior_scale)

# (optional) set the initial draw for beta
beta_hat = np.linalg.solve( np.matmul(X_t.T, X_t), np.matmul(X_t.T, y_t) )
obj.set_initial_beta_draw(beta_hat)

# %%
# (optional) set the RNG seed value of the Gibbs sampler
obj.set_seed_value(234234)
n_burnin_draws = 10_000
n_samples = 20_000
n_batches = n_samples // hN
n_keep_draws = n_batches * hN
thinning_factor = 0

beta_draws, z_draws, sigma_draws = obj.fit(p, n_burnin_draws, n_keep_draws, thinning_factor)

# %%
beta_mean = np.mean(beta_draws, axis = 1)
beta_median = np.median(beta_draws, axis=1)
beta_std = np.std(beta_draws, axis = 1)

# np.mean(z_draws, axis = 1) / np.mean(sigma_draws)

# plot beta draws

fig, axs = plt.subplots(1, 2, figsize=(15, 5), tight_layout=True)

for k in range(2):
    axs[k].hist(beta_draws[k,:], bins = 200, density=True)
    axs[k].axvline(beta_mean[k], c = 'r')
    axs[k].axvline(beta_median[k], c = 'black', ls="--")
    # dist = sts.norm(beta_mean[k], beta_std[k])
    dist = sts.t(*sts.t.fit(beta_draws[k]))
    x = np.linspace(dist.ppf(1e-6), dist.ppf(1-1e-6), num=1000)
    axs[k].plot(x, dist.pdf(x), zorder=100, c="black")
    axs[k].set_xlim(beta_mean[k] - 5 * beta_std[k], beta_mean[k] + 5 * beta_std[k])

# %%
beta_mean

# %%
b0_dist = sts.lognorm(1)
b1_dist = sts.norm(scale=0.1)

betas = np.empty((2,1))
betas[0] = b0_dist.rvs(random_state=rng)
betas[1] = b1_dist.rvs(random_state=rng)

fig, axs = plt.subplots(figsize=(12, 4), ncols=2)
_ = axs[0].plot(*_draw_pdf(b0_dist))
axs[0].axvline(betas[0], ls="--", c="red", label=np.round(betas[0], 4)[0])
axs[0].axvline(beta_mean[0], ls="--", c="black", label=np.round(beta_mean[0], 4))
_ = axs[1].plot(*_draw_pdf(b1_dist))
axs[1].axvline(betas[1], ls="--", c="red", label=np.round(betas[1], 4)[0])
axs[1].axvline(beta_mean[1], ls="--", c="black", label=np.round(beta_mean[1], 4))
axs[0].legend()
axs[0].set_title(r"$\beta_0\sim LN(1)$")
axs[1].set_title(r"$\beta_1\sim N(0, 0.1)$")
axs[1].legend()

# %%
n_post_samples = beta_draws.shape[1]
u_samples = sts.expon.rvs(scale=1, size=n_post_samples)
z_samples = sts.norm.rvs(size=n_post_samples)
post_pred = np.matmul(X, beta_draws).T + (theta * u_samples + np.sqrt(sigma_draws * u_samples) * z_samples)[:, None]

# %%
np.median(sigma_draws)

# %%
posterior = np.matmul(X, beta_draws).T
ols = np.matmul(X, beta_hat).T
posterior_exp = np.matmul(X, beta_mean).T

# %%
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(X[:, 1], y, label="observations", color="black", alpha=0.6)
ax.plot(X[:, 1], ols, c="yellow", label="OLS", alpha=0.6, ls="--")
ax.plot(X[:, 1], posterior_exp.ravel(), c="red", label=r"$\hat{q}$")
# ax.axvline(0.5, ls="--", c="black", alpha=0.3, label="in-out-sample")
ax.axvline(0.25, ls="dotted", c="black", alpha=0.5)
ax.set_title(r"90% HDI over $\hat{q}_{0.5}$")
_ = ax.legend()

# %%
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(X[:, 1], y, label="observations", color="black", alpha=0.6)
av.plot_hdi(X[:hN, 1], y=posterior[:, :hN], hdi_prob=0.9, ax=ax, color="blue", fill_kwargs={"alpha": 0.2})
av.plot_hdi(X[hN:, 1], y=posterior[:, hN:], hdi_prob=0.9, ax=ax)
ax.plot(X[:, 1], ols, c="green", label="OLS", alpha=0.6, ls="--")
ax.plot(X[:, 1], posterior_exp, c="orange", label=r"$\hat{q}$")
# ax.axvline(0.5, ls="--", c="black", alpha=0.3, label="in-out-sample")
ax.axvline(0.25, ls="dotted", c="black", alpha=0.5)
ax.set_title(r"90% HDI over $\hat{q}_{0.5}$")
ax.axhline(y[0], ls="dotted", c="black", alpha=0.3, label="zero slope")
ax.axhline(y[0], ls="dotted", c="black", alpha=0.3, label="zero slope")
ax.axhline(np.quantile(y[:hN], 0.95), xmax=X[hN, 1], label="Q first half")
ax.axhline(np.quantile(y[hN:], 0.95), xmin=X[hN, 1], label="Q second half")
_ = ax.legend()

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 5), tight_layout=True)
ax = axs[0]
_ = ax.hist(beta_draws[0] / sigma_draws, bins="auto", density=True)
ax.plot(*_draw_pdf(b0_dist), c="red", alpha=0.6)
ax.axvline(betas[0], c="red", alpha=0.6, ls="--")
ax.set_ylim(0, 15)
ax.set_xlim(-1, 5)

ax = axs[1]
_ = ax.hist(beta_draws[1] / sigma_draws, bins="auto", density=True)
ax.plot(*_draw_pdf(b1_dist), c="red", alpha=0.6)
ax.axvline(betas[1], c="red", alpha=0.6, ls="--")

# %%
plt.plot(sigma_draws)

# %%
