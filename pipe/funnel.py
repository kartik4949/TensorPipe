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


from wrappers import base_funnel
from register.register import FUNNEL

class Funnel(object):
    def __init__(self, data_path , config , datatype='bbox'):
        if datatype not in base_funnel.ALLOWED_TYPES:
            raise TypeError('datasettype not in ALLOWED_TYPEs, please check allowed dataset i.e bbox,classification labels,segmentation')
        _funnel_class = FUNNEL.get(datatype)
        self._funnel = _funnel_class(data_path,config,datatype)

    def dataset(self,type='Train'):
        if type.lower() not in ['train','val','test','validation']:
            raise Exception("Subset Data you asked is not a valid portion")

        iterable = self._funnel.dataset(type)
        return iterable

