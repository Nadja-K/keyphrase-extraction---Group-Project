#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import pke

text = u"Compatibility of systems of linear constraints over the set of natural\
 numbers. Criteria of compatibility of a system of linear Diophantine equations\
, strict inequations, and nonstrict inequations are considered. Upper bounds fo\
r components of a minimal set of solutions and algorithms of construction of mi\
nimal generating sets of solutions for all types of systems are given. These cr\
iteria and the corresponding algorithms for constructing a minimal supporting s\
et of solutions can be used in solving all the considered types systems and sys\
tems of mixed types."

grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"
pos = {'NOUN', 'PROPN', 'ADJ'}


def test_topicalpagerank_candidate_selection():
    """Test Single Topical PageRank candidate selection method."""

    extractor = pke.unsupervised.TopicalPageRank()
    extractor.load_document(input=text)
    extractor.candidate_selection(grammar=grammar)
    assert len(extractor.candidates) == 19


def test_topicalpagerank_candidate_weighting():
    """Test Single Topical PageRank weighting method."""

    extractor = pke.unsupervised.TopicalPageRank()
    extractor.load_document(input=text)
    extractor.candidate_selection(grammar=grammar)
    extractor.candidate_weighting(window=10, pos=pos)
    keyphrases = [k for k, s in extractor.get_n_best(n=3)]
    assert keyphrases == ['minimal generating sets',
                          'types systems',
                          'minimal set']

if __name__ == '__main__':
    test_topicalpagerank_candidate_selection()
    test_topicalpagerank_candidate_weighting()