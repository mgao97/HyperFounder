"""
Local data classes for DBLP4k and IMDB4k (present in dhg >= 0.9.4 but not in
the installed dhg 0.9.3). The content specs are copied verbatim from the
official dhg 0.9.4 wheel; the download/cache machinery is reused from the
installed dhg, so files land in ~/.dhg/datasets/{dblp_4k,imdb_4k}/ and are
md5-verified against the official release.
"""

from functools import partial

from dhg.datapipe import load_from_pickle, norm_ft, to_tensor, to_long_tensor
from dhg.data.base import BaseData


class DBLP4k(BaseData):
    r"""The DBLP-4k dataset (PathSim). 4,057 author vertices, 4 classes.
    Hyperedges from co-paper / co-term / co-conference correlations."""

    def __init__(self, data_root=None):
        super().__init__("dblp_4k", data_root)
        self._content = {
            'num_classes': 4,
            'num_vertices': 4057,
            'num_paper_edges': 14328,
            'num_term_edges': 7723,
            'num_conf_edges': 20,
            'dim_features': 334,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "7f8e6c3219026c284342d45c01e16406"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            'labels': {
                'upon': [{'filename': 'labels.pkl', 'md5': '6ffe5ab8c5670d8b5df595b5c4c63184'}],
                'loader': load_from_pickle,
                'preprocess': [to_long_tensor]
            },
            'edge_by_paper': {
                'upon': [{'filename': 'edge_by_paper.pkl', 'md5': 'e473eddeb4692f732bc1e47ae94d62c2'}],
                'loader': load_from_pickle,
            },
            'edge_by_term': {
                'upon': [{'filename': 'edge_by_term.pkl', 'md5': '1ca7cfbf46a7f5fc743818c65392a0ed'}],
                'loader': load_from_pickle,
            },
            'edge_by_conf': {
                'upon': [{'filename': 'edge_by_conf.pkl', 'md5': '890d683b7d8f943ac6d7e87043e0355e'}],
                'loader': load_from_pickle,
            },
            'paper_author_dict': {
                'upon': [{'filename': 'paper_author_dict.pkl', 'md5': 'eb2922e010a78961b5b66e77f9bdf950'}],
                'loader': load_from_pickle,
            },
            'term_paper_dict': {
                'upon': [{'filename': 'term_paper_dict.pkl', 'md5': '1d71f988b52b0e1da9d12f1d3fe24350'}],
                'loader': load_from_pickle,
            },
            'conf_paper_dict': {
                'upon': [{'filename': 'conf_paper_dict.pkl', 'md5': 'cbf87d64dce4ef40d2ab8406e1ee10e1'}],
                'loader': load_from_pickle,
            },
        }


class IMDB4k(BaseData):
    r"""The IMDB-4k dataset (MAGNN). 4,278 movie vertices, 3 classes.
    Hyperedges from co-actor / co-director correlations."""

    def __init__(self, data_root=None):
        super().__init__("imdb_4k", data_root)
        self._content = {
            'num_classes': 3,
            'num_vertices': 4278,
            'num_director_edges': 2081,
            'num_actor_edges': 5257,
            'dim_features': 3066,
            "features": {
                "upon": [{"filename": "features.pkl", "md5": "b9cca982d3d5066ddb2013951939c070"}],
                "loader": load_from_pickle,
                "preprocess": [to_tensor, partial(norm_ft, ord=1)],
            },
            'labels': {
                'upon': [{'filename': 'labels.pkl', 'md5': 'a45e5af53d5475ac87f5d8aa779b3a20'}],
                'loader': load_from_pickle,
                'preprocess': [to_long_tensor]
            },
            'edge_by_director': {
                'upon': [{'filename': 'edge_by_director.pkl', 'md5': '671b7c2010e8604f037523738323cd78'}],
                'loader': load_from_pickle,
            },
            'edge_by_actor': {
                'upon': [{'filename': 'edge_by_actor.pkl', 'md5': 'dff7557861445de77b05d6215746c9f1'}],
                'loader': load_from_pickle,
            },
        }
