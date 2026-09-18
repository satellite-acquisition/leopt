"""Parse CCSDS OPM, OEM, and combined NDM messages from XML.

Leaf elements are mapped to the keys used by the KVN parsers, sharing epoch
parsing and covariance conversion. Tags are matched by local name without
requiring a particular XML namespace.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

from acquisition_platform.ingest.oem import Ephemeris, _parse_oem_covariance
from acquisition_platform.ingest.opm import (
    DEFAULT_SIGMA_POS_M,
    DEFAULT_SIGMA_VEL_M_S,
    OpmState,
    _parse_covariance,
    _parse_epoch,
    _RTN_FRAME_KEYS,
)


def _local(tag: str) -> str:
    """Strip an XML namespace: '{ns}X_DOT' -> 'X_DOT'."""
    return tag.rpartition("}")[2] if "}" in tag else tag


def _find_first(root: ET.Element, name: str) -> ET.Element | None:
    """Depth-first search for the first element whose local tag == `name`."""
    name_u = name.upper()
    for el in root.iter():
        if _local(el.tag).upper() == name_u:
            return el
    return None


def looks_like_xml(text: str) -> bool:
    """Cheap sniff: does this payload start with an XML prolog / root element?"""
    s = text.lstrip()
    return s.startswith("<?xml") or s.startswith("<")


def _leaf_kv(parent: ET.Element) -> dict[str, str]:
    """Collect direct-and-nested leaf `TAG -> text` pairs under `parent`."""
    kv: dict[str, str] = {}
    for el in parent.iter():
        if el is parent:
            continue
        text = (el.text or "").strip()
        if text and len(el) == 0:  # leaf with content
            kv[_local(el.tag).upper()] = text
    return kv


def _state_vector_kv(sv: ET.Element) -> tuple[datetime, np.ndarray, np.ndarray]:
    """Parse one <stateVector> element -> (epoch, r_eci_m, v_eci_m) in SI."""
    kv = _leaf_kv(sv)
    epoch = _parse_epoch(kv["EPOCH"])
    r_km = np.array([float(kv["X"]), float(kv["Y"]), float(kv["Z"])])
    v_km_s = np.array([float(kv["X_DOT"]), float(kv["Y_DOT"]), float(kv["Z_DOT"])])
    return epoch, r_km * 1e3, v_km_s * 1e3


def parse_opm_xml(root: ET.Element) -> OpmState:
    """Parse an <opm> (or NDM child) element into an OpmState (SI units)."""
    sv = _find_first(root, "stateVector")
    if sv is None:
        raise ValueError("OPM/XML has no <stateVector> element")
    epoch, r_eci_m, v_eci_m = _state_vector_kv(sv)

    warnings: list[str] = []
    cov_el = _find_first(root, "covarianceMatrix")
    if cov_el is not None:
        cov_kv = _leaf_kv(cov_el)
        cov, have_cov = _parse_covariance(cov_kv)
        frame_kw = (cov_kv.get("COV_REF_FRAME") or "ECI").upper()
        frame = "RTN" if frame_kw in _RTN_FRAME_KEYS else "ECI"
        if not have_cov:
            frame = "ECI"
            warnings.append("OPM/XML covariance block was empty; using default sigma.")
    else:
        cov, _ = _parse_covariance({})  # -> default isotropic
        frame = "ECI"
        warnings.append(
            "OPM/XML had no covariance block; using default isotropic sigma "
            f"({DEFAULT_SIGMA_POS_M:.0f} m pos, {DEFAULT_SIGMA_VEL_M_S:.1f} m/s vel)."
        )

    return OpmState(
        epoch=epoch,
        r_eci_m=r_eci_m,
        v_eci_m=v_eci_m,
        cov_6x6=cov,
        frame=frame,
        warnings=tuple(warnings),
    )


def parse_oem_xml(root: ET.Element) -> Ephemeris:
    """Parse an <oem> (or NDM child) element into an Ephemeris (SI units)."""
    svs = [el for el in root.iter() if _local(el.tag).upper() == "STATEVECTOR"]
    if not svs:
        raise ValueError("OEM/XML has no <stateVector> elements")
    epochs: list[datetime] = []
    r_list: list[np.ndarray] = []
    v_list: list[np.ndarray] = []
    for sv in svs:
        epoch, r, v = _state_vector_kv(sv)
        epochs.append(epoch)
        r_list.append(r)
        v_list.append(v)

    order = np.argsort([e.timestamp() for e in epochs])
    epochs_sorted = tuple(epochs[i] for i in order)
    r = np.array(r_list)[order]
    v = np.array(v_list)[order]

    cov_el = _find_first(root, "covarianceMatrix")
    cov, cov_frame = (None, "ECI")
    if cov_el is not None:
        cov, cov_frame = _parse_oem_covariance(_leaf_kv(cov_el))

    meta = _find_first(root, "metadata")
    meta_kv = _leaf_kv(meta) if meta is not None else {}
    try:
        interp_degree = int(float(meta_kv.get("INTERPOLATION_DEGREE", "5")))
    except ValueError:
        interp_degree = 5

    return Ephemeris(
        epochs=epochs_sorted,
        r_eci_m=r,
        v_eci_m=v,
        object_name=meta_kv.get("OBJECT_NAME", ""),
        object_id=meta_kv.get("OBJECT_ID", ""),
        cov_6x6=cov,
        frame=cov_frame,
        interp_degree=interp_degree,
    )


def parse_ndm_xml(text: str) -> OpmState | Ephemeris | list[OpmState | Ephemeris]:
    """Parse an XML navigation message, dispatching on the message type.

    Returns an `OpmState` for an OPM, an `Ephemeris` for an OEM, and a list when
    a combined `<ndm>` wrapper carries more than one child message.
    """
    root = ET.fromstring(text)
    root_name = _local(root.tag).upper()

    if root_name == "OPM":
        return parse_opm_xml(root)
    if root_name == "OEM":
        return parse_oem_xml(root)

    # Combined NDM (or unknown root): collect every opm/oem descendant.
    results: list[OpmState | Ephemeris] = []
    for el in root.iter():
        tag = _local(el.tag).upper()
        if tag == "OPM":
            results.append(parse_opm_xml(el))
        elif tag == "OEM":
            results.append(parse_oem_xml(el))
    if not results:
        raise ValueError(f"XML root <{root_name}> contained no <opm>/<oem> messages")
    return results[0] if len(results) == 1 else results


SAMPLE_OPM_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<opm id="CCSDS_OPM_VERS" version="2.0">
  <header><CREATION_DATE>2024-01-01T00:00:00</CREATION_DATE><ORIGINATOR>PROV</ORIGINATOR></header>
  <body><segment>
    <metadata>
      <OBJECT_NAME>DEMOSAT</OBJECT_NAME>
      <OBJECT_ID>2024-001A</OBJECT_ID>
      <CENTER_NAME>EARTH</CENTER_NAME>
      <REF_FRAME>EME2000</REF_FRAME>
      <TIME_SYSTEM>UTC</TIME_SYSTEM>
    </metadata>
    <data>
      <stateVector>
        <EPOCH>2024-01-01T00:00:00.000</EPOCH>
        <X units="km">-4453.783586</X>
        <Y units="km">3995.658707</Y>
        <Z units="km">2734.025840</Z>
        <X_DOT units="km/s">-3.756048</X_DOT>
        <Y_DOT units="km/s">-2.554257</Y_DOT>
        <Z_DOT units="km/s">-6.748210</Z_DOT>
      </stateVector>
      <covarianceMatrix>
        <COV_REF_FRAME>RTN</COV_REF_FRAME>
        <CX_X>9.0e-6</CX_X>
        <CY_Y>9.0e-6</CY_Y>
        <CZ_Z>1.0e-6</CZ_Z>
        <CX_DOT_X_DOT>9.0e-12</CX_DOT_X_DOT>
        <CY_DOT_Y_DOT>9.0e-12</CY_DOT_Y_DOT>
        <CZ_DOT_Z_DOT>9.0e-12</CZ_DOT_Z_DOT>
      </covarianceMatrix>
    </data>
  </segment></body>
</opm>
"""

SAMPLE_OEM_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<oem id="CCSDS_OEM_VERS" version="2.0">
  <header><CREATION_DATE>2024-01-01T00:00:00</CREATION_DATE><ORIGINATOR>PROV</ORIGINATOR></header>
  <body><segment>
    <metadata>
      <OBJECT_NAME>DEMOSAT</OBJECT_NAME>
      <OBJECT_ID>2024-001A</OBJECT_ID>
      <CENTER_NAME>EARTH</CENTER_NAME>
      <REF_FRAME>EME2000</REF_FRAME>
      <TIME_SYSTEM>UTC</TIME_SYSTEM>
      <INTERPOLATION_DEGREE>5</INTERPOLATION_DEGREE>
    </metadata>
    <data>
      <stateVector><EPOCH>2024-01-01T00:00:00</EPOCH><X>-4453.78</X><Y>3995.66</Y><Z>2734.03</Z><X_DOT>-3.756</X_DOT><Y_DOT>-2.554</Y_DOT><Z_DOT>-6.748</Z_DOT></stateVector>
      <stateVector><EPOCH>2024-01-01T00:02:00</EPOCH><X>-4867.23</X><Y>3568.12</Y><Z>1893.40</Z><X_DOT>-3.120</X_DOT><Y_DOT>-4.560</Y_DOT><Z_DOT>-6.190</Z_DOT></stateVector>
      <stateVector><EPOCH>2024-01-01T00:04:00</EPOCH><X>-5150.00</X><Y>2990.00</Y><Z>980.00</Z><X_DOT>-1.980</X_DOT><Y_DOT>-5.980</Y_DOT><Z_DOT>-5.020</Z_DOT></stateVector>
    </data>
  </segment></body>
</oem>
"""


if __name__ == "__main__":
    assert looks_like_xml(SAMPLE_OPM_XML) and not looks_like_xml("X = 1 [km]")

    opm = parse_ndm_xml(SAMPLE_OPM_XML)
    assert isinstance(opm, OpmState)
    assert opm.frame == "RTN"
    assert abs(np.sqrt(opm.cov_6x6[0, 0]) - 3.0) < 1e-6  # sqrt(9e-6 km^2) = 3 m
    assert np.allclose(opm.cov_6x6, opm.cov_6x6.T)

    eph = parse_ndm_xml(SAMPLE_OEM_XML)
    assert isinstance(eph, Ephemeris)
    assert eph.n == 3 and eph.object_id == "2024-001A"
    r0, _ = eph.at(eph.epochs[0])
    assert np.allclose(r0, eph.r_eci_m[0], atol=1e-3)

    # Combined NDM wrapper -> a list of both.
    ndm = f"<ndm>{SAMPLE_OPM_XML.split('?>', 1)[1]}{SAMPLE_OEM_XML.split('?>', 1)[1]}</ndm>"
    both = parse_ndm_xml(ndm)
    assert isinstance(both, list) and len(both) == 2

    print(f"[ndm] OPM frame={opm.frame}, OEM N={eph.n}, NDM children={len(both)}")
    print("[ndm] PASS")
