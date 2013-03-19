from distutils.core import setup, Extension

## Remove -Wstrict-prototypes from compiler args.
## http://stackoverflow.com/questions/8106258/cc1plus-warning-command-line-option-wstrict-prototypes-is-valid-for-ada-c-o
import os
from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)
## ===

module1 = Extension('venatexll',
                    sources = ['venatexll_module.cpp'])

setup (name = 'VenationExtraction',
       version = '1.0',
       description = 'EM+MRF extraction of insect wing venation.',
       ext_modules = [module1])
