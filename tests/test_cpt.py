# import pytest
from funcsim import (
    cpt, utilPower, utilNormLog, weightTK, weightPrelec1, weightPrelec2
)
import numpy as np

def test_tk_1992_example():
    PROB = 0.16666666667
    result = cpt(
        utilFunc=lambda x: x,
        weightFuncGains=lambda p: p,
        weightFuncLosses=lambda p: p,
        refOutcome=0,
        outcomes=[-5.0, -1.0, 6.0, -3.0, 4.0, 2.0],
        probabilities=6 * [PROB]
    )
    assert isinstance(result.ExpectedValue, float)
    assert isinstance(result.CertaintyEquiv, float)
    # CptResult(ExpectedValue=0.50000000001, CertaintyEquiv=0.50000000001)
    np.testing.assert_allclose(result.ExpectedValue, 0.5, rtol=1e-9)
    np.testing.assert_allclose(result.CertaintyEquiv, 0.5, rtol=1e-9)

def test_aleks_example():
    result = cpt(
        utilFunc=lambda x: utilPower(x, alpha=0.88, beta=0.88, lamb=2.25),
        weightFuncGains=lambda p: weightPrelec1(p, gamma=0.61),
        weightFuncLosses=lambda p: weightPrelec1(p, gamma=0.69),
        refOutcome=0.0,
        outcomes=[-2, 15, 7, -5],
        probabilities=[0.3, 0.15, 0.35, 0.2]
    )
    assert isinstance(result.ExpectedValue, float)
    assert isinstance(result.CertaintyEquiv, float)
    # CptResult(ExpectedValue=0.5142746670673448, CertaintyEquiv=4.733729209057415)
    np.testing.assert_allclose(result.ExpectedValue, 0.5142746670673448, rtol=1e-9)
    np.testing.assert_allclose(result.CertaintyEquiv, 4.733729209057415, rtol=1e-9)

def test_no_probs_nonunique_outcomes_nonzero_ref():
    result = cpt(
        utilFunc=lambda x: utilPower(x, alpha=0.88, beta=0.88, lamb=2.25),
        weightFuncGains=lambda p: weightTK(p, gamma=0.61),
        weightFuncLosses=lambda p: weightTK(p, gamma=0.69),
        refOutcome=10.00,
        outcomes=[8.0, 8.0, 13.2, 17.5, 10.05]
    )
    assert isinstance(result.ExpectedValue, float)
    assert isinstance(result.CertaintyEquiv, float)
    # CptResult(ExpectedValue=0.22542186628262106, CertaintyEquiv=12.135541394014753)
    np.testing.assert_allclose(result.ExpectedValue, 0.22542186628262106, rtol=1e-9)
    np.testing.assert_allclose(result.CertaintyEquiv, 12.135541394014753, rtol=1e-9)

def test_bouchoiuicha_vieider_2017():
    result = cpt(
        utilFunc=lambda x: utilNormLog(x, gamma=1.223, delta=0.0, lamb=2.25),
        weightFuncGains=lambda p: weightPrelec2(p, alpha=0.53, beta=0.969),
        weightFuncLosses=lambda p: weightPrelec2(p, alpha=0.623, beta=0.953),
        refOutcome=100000.0,
        outcomes=[70000.0, 80000.0, 90000.0, 100000.0,
                  110000.0, 120000.0, 130000.0]
    )
    assert isinstance(result.ExpectedValue, float)
    assert isinstance(result.CertaintyEquiv, float)
    # CptResult(ExpectedValue=-22352.784861502496, CertaintyEquiv=110125.74923511136)
    np.testing.assert_allclose(result.ExpectedValue, -22352.784861502496, rtol=1e-9)
    np.testing.assert_allclose(result.CertaintyEquiv, 110125.74923511136, rtol=1e-9)

def test_no_probs_nonunique_outcomes_nonzero_ref_2():
    result = cpt(
        utilFunc=lambda x: utilPower(x, alpha=0.88, beta=0.88, lamb=2.25),
        weightFuncGains=lambda p: weightTK(p, gamma=0.61),
        weightFuncLosses=lambda p: weightTK(p, gamma=0.69),
        refOutcome=100000.0,
        outcomes=[70000.0, 80000.0, 90000.0, 100000.0,
                  110000.0, 120000.0, 130000.0]
    )
    assert isinstance(result.ExpectedValue, float)
    assert isinstance(result.CertaintyEquiv, float)
    # CptResult(ExpectedValue=-3563.361982439633, CertaintyEquiv=103982.99776749458)
    np.testing.assert_allclose(result.ExpectedValue, -3563.361982439633, rtol=1e-9)
    np.testing.assert_allclose(result.CertaintyEquiv, 103982.99776749458, rtol=1e-9)

def test_no_probs_nonunique_outcomes_nonzero_ref_3():
    result = cpt(
        utilFunc=lambda x: utilPower(x, alpha=0.88, beta=0.88, lamb=2.25),
        weightFuncGains=lambda p: weightTK(p, gamma=0.61),
        weightFuncLosses=lambda p: weightTK(p, gamma=0.69),
        refOutcome=100.0,
        outcomes=[70.0, 80.0, 90.0, 100.0,
                  110.0, 120.0, 130.0]
    )
    assert isinstance(result.ExpectedValue, float)
    assert isinstance(result.CertaintyEquiv, float)
    # CptResult(ExpectedValue=-8.163190700673397, CertaintyEquiv=103.9829977674946)
    np.testing.assert_allclose(result.ExpectedValue, -8.163190700673397, rtol=1e-9)
    np.testing.assert_allclose(result.CertaintyEquiv, 103.9829977674946, rtol=1e-9)