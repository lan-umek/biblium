# -*- coding: utf-8 -*-
"""
Numerični regresijski testi za bibliometrične indekse in zakone (2.18).

Vrednosti so izračunane ročno po definicijah iz literature. Ti testi bi
ujeli napačni implementaciji tapered_h_index in pi_index iz verzije 2.16.
"""
import math

import numpy as np
import pytest

from biblium.utilsbib import (
    h_index,
    g_index,
    hg_index,
    a_index,
    r_index,
    chi_index,
    h2_index,
    tapered_h_index,
    pi_index,
    gini_index,
)

# Testni vektor: sortiran padajoče, ročno preverljiv
CITS = [10, 8, 5, 4, 3, 2, 1, 0]


class TestCoreIndices:
    def test_h_index(self):
        # ranki 1..8: 10>=1, 8>=2, 5>=3, 4>=4, 3>=5 ne -> h=4
        assert h_index(CITS) == 4

    def test_g_index(self):
        # kumulative: 10,18,23,27,30,32,33,33; g^2: 25<=30 (g=5), 36>32 -> g=5
        assert g_index(CITS) == 5

    def test_hg_index(self):
        assert hg_index(CITS) == pytest.approx(math.sqrt(4 * 5))

    def test_a_index(self):
        # povprečje h-jedra: (10+8+5+4)/4
        assert a_index(CITS) == pytest.approx(6.75)

    def test_r_index(self):
        # sqrt vsote h-jedra: sqrt(27)
        assert r_index(CITS) == pytest.approx(math.sqrt(27))

    def test_chi_index(self):
        # sqrt(max(rang*cit)) = sqrt(max(10,16,15,16,15,12,7,0)) = 4
        assert chi_index(CITS) == pytest.approx(4.0)

    def test_h2_index_kosmulski(self):
        # rang r: c >= r^2: 10>=1, 8>=4, 5>=9 ne -> h(2)=2
        assert h2_index(CITS) == 2

    def test_pi_index_vinkler(self):
        # elitni set = int(sqrt(8)) = 2 članka; (10+8)/100 = 0.18
        assert pi_index(CITS) == pytest.approx(0.18)


class TestTaperedH:
    """Anderson, Hankin & Killworth (2008)."""

    def test_single_paper_one_citation(self):
        assert tapered_h_index([1]) == pytest.approx(1.0)

    def test_single_paper_two_citations(self):
        assert tapered_h_index([2]) == pytest.approx(1 + 1 / 3)

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_complete_ferrers_square(self, n):
        # popolnoma zapolnjen h-kvadrat vrne natanko h
        assert tapered_h_index([n] * n) == pytest.approx(float(n))

    def test_reference_implementation(self):
        # neodvisna referenčna implementacija po definiciji
        def reference(cits):
            cits = sorted((c for c in cits if c), reverse=True)
            total = 0.0
            for j, c in enumerate(cits, 1):
                if c <= j:
                    total += c / (2 * j - 1)
                else:
                    total += j / (2 * j - 1) + sum(
                        1.0 / (2 * i - 1) for i in range(j + 1, c + 1)
                    )
            return total

        for cits in ([10, 8, 5, 4, 3, 2, 1, 0], [25, 8, 5, 3, 3, 3, 2, 1], [7], [3, 3]):
            assert tapered_h_index(cits) == pytest.approx(reference(cits))

    def test_monotonicity(self):
        # dodatna citacija ne sme znižati indeksa
        assert tapered_h_index([10, 8, 6]) >= tapered_h_index([10, 8, 5])


class TestGini:
    def test_equal_distribution_is_zero(self):
        assert gini_index([5, 5, 5, 5]) == pytest.approx(0.0, abs=1e-6)

    def test_inequality_increases(self):
        assert gini_index([100, 0, 0, 0]) > gini_index([30, 30, 20, 20])

    def test_bounds(self):
        g = gini_index([50, 10, 5, 1, 0])
        assert 0.0 <= g <= 1.0


class TestEdgeCases:
    def test_empty_input(self):
        assert h_index([]) == 0

    def test_all_zeros(self):
        assert h_index([0, 0, 0]) == 0
        assert tapered_h_index([0, 0, 0]) == pytest.approx(0.0)

    def test_none_values_ignored(self):
        assert h_index([10, None, 8, None, 5, 4]) == h_index([10, 8, 5, 4])
