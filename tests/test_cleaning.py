"""Every cleaning rule is exercised on a hand-made file with one deliberate defect per row."""
import pandas as pd

from smartbin import cleaning, features, io, tables
from smartbin.config import load_config

HEADER = ("deviceid,weighttime,minwaittime,sensitivity,delay,weight,initiatortype,weightclassification,"
          "weight_bin,section_id,is_relevant_section,relevant_for_weight_analysis")
ROWS = [
    "D1,2022-03-02 10:00:00.000,1.0,8.0,30.0,1000.0,AUTOMATIC,OK,803 - 1877,S1,True,False",          # clean
    "D1,2022-03-02 10:00:00.000,1.0,8.0,30.0,1000.0,AUTOMATIC,OK,803 - 1877,S1,True,False",          # A2 duplicate
    "D1,2022-02-20 10:00:00.000,1.0,8.0,30.0,1000.0,AUTOMATIC,OK,803 - 1877,S0,True,False",          # A4 too early
    "D1,2022-03-02 11:00:00.000,1.0,8.0,30.0,-5.0,AUTOMATIC,OK,0 - 803,S1,True,False",                # A5 weight
    "D1,2022-03-02 12:00:00.000,1.0,8.0,30.0,990.0,AUTOMATIC,,803 - 1877,S1,True,False",             # B1 no label
    "D1,2022-03-02 13:00:00.000,1.0,8.0,30.0,990.0,AUTOMATIC,TOO_FEW,803 - 1877,S1,True,False",      # B2 impossible
    "D1,2022-03-02 14:00:00.000,,,,980.0,AUTOMATIC,OK,803 - 1877,,,False",                            # B3 no sens
    "D1,2022-03-02 15:00:00.000,60.0,8.0,30.0,980.0,AUTOMATIC,TOO_MANY,803 - 1877,S1,True,False",    # B4 config
    "D2,2022-03-02 15:00:00.000,1.0,8.0,30.0,980.0,AUTOMATIC,TOO_MANY,803 - 1877,S2,True,False",     # B5 no meta
    "D1,2022-03-02 16:00:00.000,1.0,8.0,30.0,980.0,MANUAL,OK,803 - 1877,S9,False,False",             # B6 not relevant
    "D1,2022-03-02 17:00:00.000,1.0,3.0,30.0,970.0,AUTOMATIC,TOO_MANY,803 - 1877,S3,True,False",     # B7 anomaly
]


def test_rules(tmp_path):
    (tmp_path / "m.csv").write_text("\n".join([HEADER] + ROWS))
    pd.DataFrame({"deviceid": ["D1", "D2"], "SCALE_NAME": ["Cloud-V1.00-6kg", None], "BOX_NAME": ["SK 2311", None],
                  "CUSTOMER_NUMBER": ["S1", None], "LOCATIONID": ["1234567.0", None],
                  "SUPPLIER_ITEM_NUMBER": ["1", None], "DESCRIPTION": ["Washer", None], "DIMENSION": ["M3", None],
                  "ITEM_WEIGHT": ["0.5", None]}).to_csv(tmp_path / "meta.csv", index=False)
    pd.DataFrame({"deviceid": ["D1", "D2"], "device_changes_sensitivity": [False, False],
                  "device_changes_weight_bin": [False, False],
                  "device_has_relevant_sections": [True, True]}).to_csv(tmp_path / "pre.csv", index=False)
    cfg = load_config(**{"paths.interim": str(tmp_path / "interim"), "paths.reports": str(tmp_path / "rep")})
    cfg.cleaning.min_devices_per_level = 1
    con = io.connect(cfg)
    io.load_measurements(con, tmp_path / "m.csv")
    fa = cleaning.apply_stage_a(con, cfg).set_index("rule").rows_removed
    assert fa["A2_duplicate"] == 1 and fa["A4_out_of_window"] == 1 and fa["A5_weight_invalid"] == 1
    meta = io.load_metadata(tmp_path / "meta.csv")
    assert meta.location_id.iloc[0] == "1234567"          # float text -> clean string id
    features.build_events(con, meta, io.load_preprocessing(tmp_path / "pre.csv"))
    fb = cleaning.apply_stage_b(con, cfg).set_index("rule").rows_removed
    for rule in ["B1_label_missing", "B2_label_inconsistent", "B3_sensitivity_missing",
                 "B4_trigger_config_nonstandard", "B5_device_metadata_missing", "B6_section_not_relevant",
                 "B7_sensitivity_anomaly"]:
        assert fb[rule] == 1, rule
    # flagged rows are NOT deleted: the event chain keeps all 8 stage-A survivors
    assert con.execute("SELECT count(*) FROM events").fetchone()[0] == 8
    assert con.execute("SELECT count(*) FROM events WHERE exclusion_reason IS NULL").fetchone()[0] == 1
    checks = tables.verify_dataset(con, cfg)
    assert (checks.check.str.contains("label logic")).any()
