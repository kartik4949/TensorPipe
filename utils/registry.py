"""
   Copyright 2020 Kartik Sharma

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import inspect
import warnings
from functools import partial


class Registry:
    """Registry.
            Registry Class which stores module references which can be used to
            apply pluging architecture and achieve flexiblity.
    """

    def __init__(self, name):
        """__init__.

        Args:
            name:
        """
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        """__len__.
        """
        return len(self._module_dict)

    def __contains__(self, key):
        """__contains__.

        Args:
            key:
        """
        return self.get(key) is not None

    def __repr__(self):
        """__repr__.
        """
        format_str = (
            self.__class__.__name__ + f"(name={self._name}, "
            f"items={self._module_dict})"
        )
        return format_str

    @property
    def name(self):
        """name.
        """
        return self._name

    @property
    def module_dict(self):
        """module_dict.
        """
        return self._module_dict

    def get(self, key):
        """get.

        Args:
            key:
        """
        return self._module_dict.get(key, None)

    def _register_module(self, module_class, module_name=None, force=False):
        """_register_module.

        Args:
            module_class: Module class to register
            module_name: Module name to register
            force: forced injection in register
        """

        if module_name is None:
            module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError(
                f"{module_name} is already registered " f"in {self.name}"
            )
        self._module_dict[module_name] = module_class

    def register_module(self, name=None, force=False, module=None):
        """register_module.
            Registers module passed and stores in the modules dict.

        Args:
            name: module name.
            force: if forced inject register module if already present. default False.
            module: Module Reference.
        """

        if module is not None:
            self._register_module(
                module_class=module, module_name=name, force=force
            )
            return module

        if not (name is None or isinstance(name, str)):
            raise TypeError(f"name must be a str, but got {type(name)}")

        def _register(cls):
            self._register_module(
                module_class=cls, module_name=name, force=force
            )
            return cls

        return _register
