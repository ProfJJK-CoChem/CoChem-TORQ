import sys
sys.path.append(r"D:\__CoChem\GitHub-Repo\CoChem-TORQ")
from Libraries.cochem_tensor_extractor import get_atomic_mass

def test_isotopic_mass_table_accuracy_and_parsing() -> None:
    """Validates mono-isotopic mass lookups via mendeleev."""
    import mendeleev
    
    # Key isotopes
    # H
    elem_h = mendeleev.element("H")
    h1_mass = float(next(i for i in elem_h.isotopes if i.mass_number == 1).mass)
    h2_mass = float(next(i for i in elem_h.isotopes if i.mass_number == 2).mass)
    h3_mass = float(next(i for i in elem_h.isotopes if i.mass_number == 3).mass)
    assert abs(get_atomic_mass("H") - h1_mass) < 1e-8
    assert abs(get_atomic_mass("1H") - h1_mass) < 1e-8
    assert abs(get_atomic_mass("D") - h2_mass) < 1e-8
    assert abs(get_atomic_mass("2H") - h2_mass) < 1e-8
    assert abs(get_atomic_mass("T") - h3_mass) < 1e-8
    assert abs(get_atomic_mass("3H") - h3_mass) < 1e-8

    # C
    elem_c = mendeleev.element("C")
    most_abundant_c = sorted([i for i in elem_c.isotopes if i.abundance is not None], key=lambda x: x.abundance, reverse=True)[0]
    c_mass = float(most_abundant_c.mass)
    c12_mass = float(next(i for i in elem_c.isotopes if i.mass_number == 12).mass)
    c13_mass = float(next(i for i in elem_c.isotopes if i.mass_number == 13).mass)
    
    assert abs(get_atomic_mass("C") - c_mass) < 1e-8
    assert abs(get_atomic_mass("12C") - c12_mass) < 1e-8
    assert abs(get_atomic_mass("13C") - c13_mass) < 1e-8
    assert abs(get_atomic_mass("C13") - c13_mass) < 1e-8

    # N
    elem_n = mendeleev.element("N")
    most_abundant_n = sorted([i for i in elem_n.isotopes if i.abundance is not None], key=lambda x: x.abundance, reverse=True)[0]
    n_mass = float(most_abundant_n.mass)
    n15_mass = float(next(i for i in elem_n.isotopes if i.mass_number == 15).mass)
    assert abs(get_atomic_mass("N") - n_mass) < 1e-8
    assert abs(get_atomic_mass("15N") - n15_mass) < 1e-8

    # Fallback to default for unrecognized elements
    assert get_atomic_mass("UnknownElement") == 12.0
    print("Test passed!")

if __name__ == "__main__":
    test_isotopic_mass_table_accuracy_and_parsing()
