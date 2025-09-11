"""
PyInstaller hook for scikit-optimize (skopt)

This hook ensures that all necessary components of scikit-optimize are properly included.
"""

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

# Collect everything from skopt
datas, binaries, hiddenimports = collect_all('skopt')

# Also collect submodules explicitly
hiddenimports += collect_submodules('skopt')

# Ensure key dependencies are included
hiddenimports += [
    'skopt.acquisition',
    'skopt.benchmarks', 
    'skopt.callbacks',
    'skopt.learning',
    'skopt.learning.forest',
    'skopt.learning.gbrt',
    'skopt.learning.gaussian_process',
    'skopt.learning.gaussian_process.gpr',
    'skopt.learning.gaussian_process.kernels',
    'skopt.optimizer',
    'skopt.optimizer.optimizer',
    'skopt.optimizer.base',
    'skopt.optimizer.forest',
    'skopt.optimizer.gbrt',
    'skopt.optimizer.gp',
    'skopt.optimizer.dummy',
    'skopt.space',
    'skopt.space.space',
    'skopt.space.transformers',
    'skopt.utils',
    'skopt.plots',
    'skopt.sampler',
    'skopt.sampler.halton',
    'skopt.sampler.sobol',
    'skopt.sampler.lhs',
    'skopt.sampler.grid',
    'skopt.sampler.hammersly',
    'skopt._hessian_update_strategy',
    'skopt._minimum',
    
    # Ensure sklearn dependencies are included
    'sklearn',
    'sklearn.base',
    'sklearn.ensemble',
    'sklearn.ensemble._forest',
    'sklearn.tree',
    'sklearn.tree._tree',
    'sklearn.gaussian_process',
    'sklearn.gaussian_process.kernels',
    'sklearn.utils',
    'sklearn.utils.validation',
    'sklearn.utils.optimize',
    'sklearn.preprocessing',
    'sklearn.model_selection',
    
    # Scipy dependencies
    'scipy.optimize',
    'scipy.optimize._minimize',
    'scipy.optimize._lbfgsb',
    'scipy.optimize.optimize',
    'scipy.stats',
    'scipy.stats.distributions',
    'scipy.spatial',
    'scipy.spatial.distance',
    
    # Numpy dependencies
    'numpy',
    'numpy.random',
    'numpy.linalg',
    
    # Joblib for parallel processing
    'joblib',
    'joblib.parallel',
]