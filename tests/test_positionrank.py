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


def test_positionrank_candidate_selection():
    """Test PositionRank candidate selection method."""

    extractor = pke.unsupervised.PositionRank()
    extractor.load_document(input=text)
    extractor.candidate_selection(grammar=grammar)
    assert len(extractor.candidates) == 19


def test_positionrank_candidate_weighting():
    """Test SingleRank candidate weighting method."""

    extractor = pke.unsupervised.PositionRank()
    extractor.load_document(input=text)
    extractor.candidate_selection(grammar=grammar)
    extractor.candidate_weighting(window=10, pos=pos)
    keyphrases = [k for k, s in extractor.get_n_best(n=3)]
    assert keyphrases == ['types systems',
                          'minimal generating sets',
                          'linear diophantine equations']


if __name__ == '__main__':
    test_positionrank_candidate_selection()
    test_positionrank_candidate_weighting()