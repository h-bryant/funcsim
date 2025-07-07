import numpy as np
import pandas as pd
import funcsim as fs


def test_cg_0():
    udata = np.random.random(size=(1000, 2))
    udataPd = pd.DataFrame(udata, columns=["rain", "temp"])
    cg = fs.CopulaGauss(udataPd)

    def f(ugen):
        draw = cg.draw(ugen)
        return {"rain": draw.rain, "temp": draw.temp}

    sampl = fs.simulate(f=f, ntrials=2000).sel(steps=0).values
    assert sampl.shape == (2000, 2)
    u_test_results = fs.utests(sampl[:,0])
    assert u_test_results["anderson_darling_pval"] > 0.05


def test_cg_1():
    udata = np.random.random(size=(1000, 2))
    cg = fs.CopulaGauss(udata)

    def f(ugen):
        draw = cg.draw(ugen)
        return {"rain": draw.v0, "temp": draw.v1}

    sampl = fs.simulate(f=f, ntrials=2000).sel(steps=0).values
    assert sampl.shape == (2000, 2)
    u_test_results = fs.utests(sampl[:,0])
    assert u_test_results["anderson_darling_pval"] > 0.05


def test_ct_1():
    udata = np.random.random(size=(1000, 2))
    ct = fs.CopulaStudent(udata)

    def f(ugen):
        draw = ct.draw(ugen)
        return {"rain": draw.v0, "temp": draw.v1}

    sampl = fs.simulate(f=f, ntrials=2000).sel(steps=0).values
    assert sampl.shape == (2000, 2)
    u_test_results = fs.utests(sampl[:,0])
    assert u_test_results["anderson_darling_pval"] > 0.05
