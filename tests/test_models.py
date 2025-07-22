import importlib.util
import subprocess
import numpy as np
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / 'Econometry.8.py'
spec = importlib.util.spec_from_file_location('econometry', MODULE_PATH)
module = importlib.util.module_from_spec(spec)
import sys
sys.modules['econometry'] = module
spec.loader.exec_module(module)


def test_load_csv(tmp_path):
    path = tmp_path / 'data.csv'
    path.write_text('P,Q\n1,2\n2,4\n')
    prices, qty = module.load_csv_data(str(path))
    assert np.allclose(prices, [1, 2])
    assert np.allclose(qty, [2, 4])


def test_linear_fit():
    x = np.array([1, 2, 3, 4])
    y = 2 * x + 1
    result = module.fit_model(x, y, 'linear')
    assert np.allclose(result.params, [2, 1], atol=1e-2)
    assert result.r2 > 0.99


def test_elasticity():
    func = lambda p: 2 * p + 1
    e = module.numerical_elasticity(func, 2)
    assert abs(e - 0.8) < 1e-3


def test_cli_help():
    out = subprocess.run(['python3', str(MODULE_PATH), '--help'], capture_output=True)
    assert out.returncode == 0
    assert b'--model' in out.stdout
