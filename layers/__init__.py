# -*- coding: utf-8 -*-

from .aggregator1 import SumAggregator1, ConcatAggregator1, NeighAggregator1

Aggregator1 = {
    'sum': SumAggregator1,
    'concat': ConcatAggregator1,
    'neigh': NeighAggregator1
}
