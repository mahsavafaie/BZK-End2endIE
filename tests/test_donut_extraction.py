from inferable.models.utils import extract_info

def test_extraction():
    text = "<s_BZKNr>bla<s_BZKNr>8307/IV/311</s_BZKNr>blub</s_BZKNr>"
    extract_info(text, "BZKNr", smallest_distance=True)

    assert(extract_info(text, "BZKNr", smallest_distance=True) == "8307/IV/311")
    assert(extract_info(text, "BZKNr", smallest_distance=False, remove_tags_inside=False) == "bla<s_BZKNr>8307/IV/311</s_BZKNr>blub")
    assert(extract_info(text, "BZKNr", smallest_distance=False, remove_tags_inside=True) == "bla8307/IV/311blub")

def test_partial_extraction():
    text = "<s_BZKNr>abc</s_bar>"
    assert(extract_info(text, "BZKNr", smallest_distance=True, allow_partial_match=False) == "")
    assert(extract_info(text, "BZKNr", smallest_distance=True, allow_partial_match=True) == "abc")

    text = ">abc</s_BZKNr>"
    assert(extract_info(text, "BZKNr", smallest_distance=True, allow_partial_match=False) == "")
    assert(extract_info(text, "BZKNr", smallest_distance=True, allow_partial_match=True) == "abc")

    text = "abc</s_BZKNr>"
    assert(extract_info(text, "BZKNr", smallest_distance=True, allow_partial_match=False) == "")
    assert(extract_info(text, "BZKNr", smallest_distance=True, allow_partial_match=True) == "abc")

    text = "<s_BZKNr>abc"
    assert(extract_info(text, "BZKNr", smallest_distance=True, allow_partial_match=False) == "")
    assert(extract_info(text, "BZKNr", smallest_distance=True, allow_partial_match=True) == "abc")

def test_partial_extraction_two():
    text ="<s_BZKNr> 2/ 01884</s_BZKNr><s_ApplicantFirstName> Josef</s_ApplicantCurrentAddress>"
    assert(extract_info(text, "ApplicantFirstName", allow_partial_match=True) == "Josef")