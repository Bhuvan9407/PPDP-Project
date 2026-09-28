\
    from pathlib import Path
    import sys
    import subprocess
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[1]

    print("=" * 70)
    print("KC-SLICE MILESTONE 2 — SETUP VERIFICATION")
    print("=" * 70)

    # Adult
    adult_path = ROOT / "data/raw/adult.csv"
    if not adult_path.exists():
        raise FileNotFoundError(f"Missing: {adult_path}")

    adult = pd.read_csv(adult_path)
    print("✅ Adult")
    print(f"   Rows    : {len(adult)}")
    print(f"   Columns : {len(adult.columns)}")
    assert len(adult) == 32561
    assert len(adult.columns) == 15

    # Diabetes
    diabetes_path = ROOT / "data/raw/diabetic_data.csv"
    if not diabetes_path.exists():
        raise FileNotFoundError(f"Missing: {diabetes_path}")

    diabetes = pd.read_csv(diabetes_path)
    print()
    print("✅ Diabetes")
    print(f"   Rows    : {len(diabetes)}")
    print(f"   Columns : {len(diabetes.columns)}")
    assert len(diabetes) == 101766

    print("   '?' sentinel verified in:")
    for col in ["race", "weight", "payer_code", "medical_specialty"]:
        assert (diabetes[col].astype(str) == "?").any(), col
        print(f"      {col}")

    # Smoke files
    smoke_adult = ROOT / "data/smoke/adult_small.csv"
    smoke_diabetes = ROOT / "data/smoke/diabetes_small.csv"

    if smoke_adult.exists() and smoke_diabetes.exists():
        a = pd.read_csv(smoke_adult)
        d = pd.read_csv(smoke_diabetes)
        assert len(a) == 100
        assert len(d) == 100
        print()
        print("✅ Smoke datasets")
        print(f"   Adult     : {a.shape}")
        print(f"   Diabetes  : {d.shape}")
    else:
        print()
        print("ℹ️ Smoke datasets not found; run the Colab smoke-data cell first.")

    # Mondrian import
    mondrian_path = str(ROOT / "methods/mondrian/upstream")
    if Path(mondrian_path).exists():
        sys.path.insert(0, mondrian_path)
        import mondrian
        print()
        print("✅ Mondrian")
        print(f"   Module: {mondrian.__file__}")
    else:
        print()
        print("ℹ️ Mondrian upstream directory not found.")

    # ARX JAR
    arx_jar = ROOT / "methods/arx/libarx-3.9.2.jar"
    if arx_jar.exists():
        print()
        print("✅ ARX")
        print(f"   JAR: {arx_jar}")
    else:
        print()
        print("ℹ️ ARX JAR not found.")

    print()
    print("=" * 70)
    print("MILESTONE 2 REPOSITORY VERIFICATION COMPLETE")
    print("=" * 70)
