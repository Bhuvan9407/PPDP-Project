"""
MILESTONE 3 BASELINE REPRODUCTION SOURCE

This file is an auditable source extraction from the executed
Milestone 3 Colab notebook.

The code below is preserved from the notebook execution history
rather than rewritten. The source cell numbers are included so
the implementation can be traced back to the experimental run.
"""


########################################################################
# SOURCE NOTEBOOK EXECUTION CELL 24
########################################################################

#cell 24
# ============================================================
# MILESTONE 3 — STANDARD ADULT/MONDRIAN BASELINE CONFIG
# ============================================================

from pathlib import Path
import pandas as pd

ROOT = Path("/content/kc-slice")

K = 10

QID_COLUMNS = [
    "age",
    "workclass",
    "education.num",
    "marital.status",
    "occupation",
    "race",
    "sex",
    "native.country",
]

SENSITIVE_COLUMN = "income"

print("=" * 70)
print("MILESTONE 3 — STANDARD ADULT BASELINE")
print("=" * 70)

print(f"k                   : {K}")
print("QIDs                :")
for i, col in enumerate(QID_COLUMNS, 1):
    print(f"  {i}. {col}")

print(f"Sensitive attribute : {SENSITIVE_COLUMN}")

# Load Adult
adult_raw = pd.read_csv(
    ROOT / "data/raw/adult.csv"
)

print()
print("Raw Adult dataset")
print(f"Rows    : {len(adult_raw)}")
print(f"Columns : {len(adult_raw.columns)}")

# Remove rows containing '?'
adult_clean = adult_raw[
    ~(adult_raw.astype(str) == "?").any(axis=1)
].copy()

print()
print("After removing records containing '?'")
print(f"Rows    : {len(adult_clean)}")
print(f"Columns : {len(adult_clean.columns)}")

assert len(adult_clean) == 30162

# Select standard Mondrian configuration
baseline_columns = QID_COLUMNS + [SENSITIVE_COLUMN]

adult_baseline = adult_clean[baseline_columns].copy()

output_path = (
    ROOT / "data/processed/adult_standard_baseline_k10_1sa.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

adult_baseline.to_csv(
    output_path,
    index=False
)

print()
print("Baseline dataset")
print(f"Shape : {adult_baseline.shape}")
print(f"Saved : {output_path}")

print()
print(adult_baseline.head())

print()
print("=" * 70)
print("STANDARD ADULT BASELINE PREPARATION PASSED")
print("=" * 70)

########################################################################
# SOURCE NOTEBOOK EXECUTION CELL 25
########################################################################

# Cell 25
# ============================================================
# MILESTONE 3 — STANDARD ADULT MONDRIAN BASELINE
# 8 QIDs, 1 sensitive attribute, k=10
# ============================================================

from pathlib import Path
from collections import Counter
import copy
import sys
import pandas as pd

ROOT = Path("/content/kc-slice")
MONDRIAN_PATH = ROOT / "methods/mondrian/upstream"

if str(MONDRIAN_PATH) not in sys.path:
    sys.path.insert(0, str(MONDRIAN_PATH))

from mondrian import mondrian

K = 10

QID_COLUMNS = [
    "age",
    "workclass",
    "education.num",
    "marital.status",
    "occupation",
    "race",
    "sex",
    "native.country",
]

SENSITIVE_COLUMN = "income"

# ------------------------------------------------------------
# Load standard baseline
# ------------------------------------------------------------

input_path = (
    ROOT / "data/processed/adult_standard_baseline_k10_1sa.csv"
)

df = pd.read_csv(input_path)

assert len(df) == 30162
assert all(col in df.columns for col in QID_COLUMNS)
assert SENSITIVE_COLUMN in df.columns

print("=" * 70)
print("STANDARD ADULT — MONDRIAN BASELINE")
print("=" * 70)
print(f"Input rows : {len(df):,}")
print(f"QID count  : {len(QID_COLUMNS)}")
print(f"QIDs       : {QID_COLUMNS}")
print(f"Sensitive  : {SENSITIVE_COLUMN}")
print(f"k          : {K}")

# ------------------------------------------------------------
# Encode categorical QIDs using first-appearance ordering
# ------------------------------------------------------------

NUMERIC_QIDS = ["age", "education.num"]
CATEGORICAL_QIDS = [
    col for col in QID_COLUMNS
    if col not in NUMERIC_QIDS
]

category_orders = {}
category_maps = {}

for col in CATEGORICAL_QIDS:
    order = list(dict.fromkeys(df[col].tolist()))
    category_orders[col] = order
    category_maps[col] = {
        value: index for index, value in enumerate(order)
    }

print()
print("Categorical ordering:")
for col in CATEGORICAL_QIDS:
    print(f"{col}: {category_orders[col]}")

# ------------------------------------------------------------
# Convert records to Mondrian format:
# [qid1, qid2, ..., qid8, sensitive]
# ------------------------------------------------------------

data = []

for _, row in df.iterrows():
    record = []

    for col in QID_COLUMNS:
        if col in NUMERIC_QIDS:
            record.append(int(row[col]))
        else:
            record.append(category_maps[col][row[col]])

    record.append(row[SENSITIVE_COLUMN])
    data.append(record)

assert len(data) == len(df)
assert all(len(record) == 9 for record in data)

# ------------------------------------------------------------
# Run strict Mondrian
# ------------------------------------------------------------

print()
print("Running strict Mondrian...")

result, eval_result = mondrian(
    copy.deepcopy(data),
    K,
    False
)

ncp = eval_result[0]
runtime = eval_result[1]

# ------------------------------------------------------------
# Calculate equivalence classes and DM
# ------------------------------------------------------------

equivalence_classes = Counter(
    tuple(record[:len(QID_COLUMNS)])
    for record in result
)

class_sizes = list(equivalence_classes.values())

assert class_sizes, "Mondrian returned no equivalence classes"

discernibility = sum(size ** 2 for size in class_sizes)

# ------------------------------------------------------------
# Verify output
# ------------------------------------------------------------

print()
print("=" * 70)
print("STANDARD MONDRIAN RESULTS")
print("=" * 70)

print(f"Input rows              : {len(df):,}")
print(f"Output rows             : {len(result):,}")
print(f"QID count               : {len(QID_COLUMNS)}")
print(f"NCP (%)                 : {ncp:.4f}")
print(f"Runtime (seconds)       : {runtime:.4f}")
print(f"Equivalence classes     : {len(class_sizes):,}")
print(f"Minimum class size      : {min(class_sizes):,}")
print(f"Maximum class size      : {max(class_sizes):,}")
print(f"Mean class size         : {sum(class_sizes)/len(class_sizes):.2f}")
print(f"Discernibility Metric   : {discernibility:,}")

assert len(result) == len(df), (
    f"Record count changed: {len(df)} -> {len(result)}"
)

assert min(class_sizes) >= K, (
    f"k-anonymity violation: min class size = {min(class_sizes)}"
)

assert sum(class_sizes) == len(result)

print()
print("Record accounting       : PASS")
print("k=10 anonymity          : PASS")

# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------

results_dir = ROOT / "results"
results_dir.mkdir(parents=True, exist_ok=True)

summary_path = results_dir / "milestone3_mondrian_standard_8qid.txt"

summary = f"""MILESTONE 3 — STANDARD ADULT MONDRIAN BASELINE

Dataset: Adult Census Income
Input records: {len(df)}
Output records: {len(result)}
QIDs: {", ".join(QID_COLUMNS)}
Sensitive attribute: {SENSITIVE_COLUMN}
k: {K}
Algorithm: qiyuangong/Mondrian
Model: strict

NCP (%): {ncp:.4f}
Runtime (seconds): {runtime:.4f}
Equivalence classes: {len(class_sizes)}
Minimum class size: {min(class_sizes)}
Maximum class size: {max(class_sizes)}
Mean class size: {sum(class_sizes)/len(class_sizes):.2f}
Discernibility Metric: {discernibility}

Record accounting: PASS
k-anonymity verification: PASS
"""

summary_path.write_text(summary, encoding="utf-8")

print(f"Saved summary: {summary_path}")
print()
print("=" * 70)
print("STANDARD MONDRIAN BASELINE COMPLETE")
print("=" * 70)

########################################################################
# SOURCE NOTEBOOK EXECUTION CELL 26
########################################################################

# Cell 26
# ============================================================
# MILESTONE 3 — STANDARD ADULT ARX BASELINE
# 8 QIDs, k=10
#
# This baseline:
#   - uses the same 30,162-row Adult dataset as Mondrian
#   - uses explicit hand-built hierarchies
#   - uses k-anonymity with k=10
#   - uses zero tuple suppression
#   - keeps income unchanged as a non-QID attribute
#   - verifies the materialized output independently
# ============================================================

from pathlib import Path
from collections import Counter
import subprocess
import time
import pandas as pd

ROOT = Path("/content/kc-slice")

ARX_DIR = ROOT / "methods/arx"
JAR = ARX_DIR / "libarx-3.9.2.jar"

DATA = ROOT / "data/processed/adult_standard_baseline_k10_1sa.csv"

HIER_DIR = ROOT / "hierarchies/adult_arx_standard"
OUT_DIR = ROOT / "results/arx_standard_adult"

HIER_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

JAVA_FILE = ARX_DIR / "AdultArxBaseline.java"
OUTPUT_FILE = OUT_DIR / "adult_standard_arx_k10.csv"

K = 10

QID_COLUMNS = [
    "age",
    "workclass",
    "education.num",
    "marital.status",
    "occupation",
    "race",
    "sex",
    "native.country",
]

SENSITIVE_COLUMN = "income"


# ============================================================
# 1. Verify prerequisites
# ============================================================

assert DATA.exists(), f"Missing input dataset: {DATA}"
assert JAR.exists(), f"Missing ARX JAR: {JAR}"

df = pd.read_csv(DATA)

assert len(df) == 30162, (
    f"Expected 30,162 input rows, got {len(df)}"
)

expected_columns = QID_COLUMNS + [SENSITIVE_COLUMN]

assert list(df.columns) == expected_columns, (
    "Unexpected baseline columns:\n"
    f"{df.columns.tolist()}"
)

print("=" * 70)
print("MILESTONE 3 — STANDARD ADULT ARX BASELINE")
print("=" * 70)
print(f"Input rows : {len(df):,}")
print(f"QID count  : {len(QID_COLUMNS)}")
print(f"Sensitive  : {SENSITIVE_COLUMN}")
print(f"k          : {K}")
print(f"ARX JAR    : {JAR}")


# ============================================================
# 2. Create ARX hierarchies
# ============================================================

print()
print("Creating ARX hierarchies...")


# ------------------------------------------------------------
# AGE
#
# exact age -> decade -> *
# ------------------------------------------------------------

age_hierarchy = HIER_DIR / "age.csv"

with age_hierarchy.open("w", encoding="utf-8") as f:

    for age in sorted(df["age"].astype(int).unique()):

        decade_start = (age // 10) * 10
        decade_end = min(decade_start + 9, 99)

        decade = f"{decade_start}-{decade_end}"

        f.write(f"{age},{decade},*\n")


# ------------------------------------------------------------
# EDUCATION.NUM
#
# exact value -> band -> *
# ------------------------------------------------------------

education_hierarchy = HIER_DIR / "education.num.csv"

with education_hierarchy.open("w", encoding="utf-8") as f:

    for value in sorted(df["education.num"].astype(int).unique()):

        if value <= 4:
            band = "1-4"
        elif value <= 6:
            band = "5-6"
        elif value <= 8:
            band = "7-8"
        elif value <= 10:
            band = "9-10"
        elif value <= 12:
            band = "11-12"
        elif value <= 14:
            band = "13-14"
        else:
            band = "15-16"

        f.write(f"{value},{band},*\n")


# ------------------------------------------------------------
# CATEGORICAL QIDS
#
# exact category -> *
# ------------------------------------------------------------

categorical_qids = [
    col
    for col in QID_COLUMNS
    if col not in ["age", "education.num"]
]

for col in categorical_qids:

    hierarchy_path = HIER_DIR / f"{col}.csv"

    # Preserve first-appearance order.
    values = list(
        dict.fromkeys(
            df[col].astype(str).tolist()
        )
    )

    with hierarchy_path.open("w", encoding="utf-8") as f:

        for value in values:
            f.write(f"{value},*\n")


print("Hierarchies created:")

for path in sorted(HIER_DIR.glob("*.csv")):
    print(f"  {path.name}")


# ============================================================
# 3. Generate Java ARX runner
# ============================================================

print()
print("Generating Java ARX runner...")

java_source = r'''
import java.io.File;
import java.nio.charset.StandardCharsets;

import org.deidentifier.arx.ARXAnonymizer;
import org.deidentifier.arx.ARXConfiguration;
import org.deidentifier.arx.ARXResult;
import org.deidentifier.arx.AttributeType;
import org.deidentifier.arx.Data;
import org.deidentifier.arx.criteria.KAnonymity;
import org.deidentifier.arx.metric.Metric;

public class AdultArxBaseline {

    public static void main(String[] args) throws Exception {

        if (args.length != 3) {
            System.err.println(
                "Usage: AdultArxBaseline <input.csv> <hierarchy_dir> <output.csv>"
            );
            System.exit(1);
        }

        String inputPath = args[0];
        String hierarchyDir = args[1];
        String outputPath = args[2];

        // ----------------------------------------------------
        // Load input dataset
        // ----------------------------------------------------

        System.out.println("Loading ARX input...");

        Data data = Data.create(
            inputPath,
            StandardCharsets.UTF_8,
            ','
        );

        // ----------------------------------------------------
        // QIDs and their hierarchies
        // ----------------------------------------------------

        String[] qids = {
            "age",
            "workclass",
            "education.num",
            "marital.status",
            "occupation",
            "race",
            "sex",
            "native.country"
        };

        for (String qid : qids) {

            File hierarchyFile = new File(
                hierarchyDir,
                qid + ".csv"
            );

            // Mark attribute as a quasi-identifier.
            data.getDefinition().setAttributeType(
                qid,
                AttributeType.QUASI_IDENTIFYING_ATTRIBUTE
            );

            // Attach hierarchy.
            data.getDefinition().setHierarchy(
                qid,
                AttributeType.Hierarchy.create(
                    hierarchyFile,
                    StandardCharsets.UTF_8,
                    ','
                )
            );
        }

        // ----------------------------------------------------
        // Income
        //
        // This k-anonymity baseline does not use a
        // sensitive-attribute privacy criterion.
        //
        // Therefore income is preserved unchanged and is
        // excluded from the QID equivalence relation.
        // ----------------------------------------------------

        data.getDefinition().setAttributeType(
            "income",
            AttributeType.INSENSITIVE_ATTRIBUTE
        );

        // ----------------------------------------------------
        // ARX configuration
        // ----------------------------------------------------

        ARXConfiguration config = ARXConfiguration.create();

        // k-anonymity
        config.addPrivacyModel(
            new KAnonymity(10)
        );

        // Zero suppression: every input record must remain.
        config.setSuppressionLimit(0d);

        // ARX loss metric used for optimization.
        config.setQualityModel(
            Metric.createLossMetric()
        );

        // ----------------------------------------------------
        // Run ARX
        // ----------------------------------------------------

        System.out.println("Running ARX...");

        ARXResult result =
            new ARXAnonymizer().anonymize(
                data,
                config
            );

        // ----------------------------------------------------
        // Verify ARX produced a valid optimum
        // ----------------------------------------------------

        if (!result.getOptimumFound()) {

            throw new RuntimeException(
                "ARX did not find a valid optimum."
            );
        }

        if (result.getGlobalOptimum() == null) {

            throw new RuntimeException(
                "ARX returned a null global optimum."
            );
        }

        System.out.println(
            "Global optimum found: PASS"
        );

        // ----------------------------------------------------
        // Save anonymized output
        // ----------------------------------------------------

        System.out.println(
            "Saving anonymized output..."
        );

        result
            .getOutput(false)
            .save(
                outputPath,
                ','
            );

        System.out.println(
            "Output saved: " + outputPath
        );
    }
}
'''

JAVA_FILE.write_text(
    java_source,
    encoding="utf-8"
)

print(f"Java source: {JAVA_FILE}")


# ============================================================
# 4. Compile ARX runner
# ============================================================

print()
print("Compiling ARX runner...")

compile_cmd = [
    "javac",
    "-cp",
    str(JAR),
    str(JAVA_FILE),
]

compile_result = subprocess.run(
    compile_cmd,
    capture_output=True,
    text=True
)

if compile_result.stdout.strip():
    print(compile_result.stdout)

if compile_result.stderr.strip():
    print(compile_result.stderr)

if compile_result.returncode != 0:

    raise RuntimeError(
        "ARX Java compilation failed."
    )

print("Compilation: PASS")


# ============================================================
# 5. Run ARX
# ============================================================

print()
print("Running ARX anonymization...")

run_cmd = [
    "java",
    "-Xmx4g",
    "-cp",
    f"{JAR}:{ARX_DIR}",
    "AdultArxBaseline",
    str(DATA),
    str(HIER_DIR),
    str(OUTPUT_FILE),
]

start_time = time.perf_counter()

run_result = subprocess.run(
    run_cmd,
    capture_output=True,
    text=True
)

runtime_seconds = (
    time.perf_counter() - start_time
)

if run_result.stdout.strip():
    print(run_result.stdout)

if run_result.stderr.strip():
    print("ARX stderr:")
    print(run_result.stderr)

if run_result.returncode != 0:

    raise RuntimeError(
        "ARX anonymization failed."
    )

print(
    f"Python-measured runtime (seconds): "
    f"{runtime_seconds:.4f}"
)


# ============================================================
# 6. Verify ARX output exists
# ============================================================

assert OUTPUT_FILE.exists(), (
    f"ARX output not found: {OUTPUT_FILE}"
)


# ============================================================
# 7. Load anonymized output
# ============================================================

arx_output = pd.read_csv(
    OUTPUT_FILE
)

print()
print("=" * 70)
print("ARX OUTPUT VERIFICATION")
print("=" * 70)

print(
    f"Input rows  : {len(df):,}"
)

print(
    f"Output rows : {len(arx_output):,}"
)


# ============================================================
# 8. Record accounting
# ============================================================

assert len(arx_output) == len(df), (
    "Record accounting failed: "
    f"{len(df):,} input rows -> "
    f"{len(arx_output):,} output rows"
)

print(
    "Record accounting       : PASS"
)


# ============================================================
# 9. Calculate realized equivalence classes
# ============================================================

equivalence_classes = Counter(
    tuple(row[col] for col in QID_COLUMNS)
    for _, row in arx_output.iterrows()
)

class_sizes = list(
    equivalence_classes.values()
)

assert len(class_sizes) > 0, (
    "No equivalence classes were produced."
)

discernibility = sum(
    size ** 2
    for size in class_sizes
)

mean_class_size = (
    sum(class_sizes) /
    len(class_sizes)
)

print()
print(
    f"Equivalence classes    : "
    f"{len(class_sizes):,}"
)

print(
    f"Minimum class size     : "
    f"{min(class_sizes):,}"
)

print(
    f"Maximum class size     : "
    f"{max(class_sizes):,}"
)

print(
    f"Mean class size        : "
    f"{mean_class_size:.2f}"
)

print(
    f"Discernibility Metric  : "
    f"{discernibility:,}"
)


# ============================================================
# 10. Verify k-anonymity
# ============================================================

assert min(class_sizes) >= K, (
    "k-anonymity verification failed: "
    f"minimum class size = {min(class_sizes)}"
)

assert sum(class_sizes) == len(arx_output), (
    "Equivalence-class accounting failed."
)

print(
    "k=10 anonymity          : PASS"
)


# ============================================================
# 11. Save milestone summary
# ============================================================

summary_path = (
    ROOT /
    "results" /
    "milestone3_arx_baseline.txt"
)

summary = f"""
MILESTONE 3 — STANDARD ADULT ARX BASELINE

Dataset:
  Adult Census Income

Input records:
  {len(df)}

Output records:
  {len(arx_output)}

QIDs:
{chr(10).join("  - " + q for q in QID_COLUMNS)}

Non-QID attribute:
  income

Privacy model:
  k-anonymity

k:
  {K}

Suppression limit:
  0

ARX version:
  3.9.2

Runtime (seconds):
  {runtime_seconds:.4f}

Equivalence classes:
  {len(class_sizes)}

Minimum class size:
  {min(class_sizes)}

Maximum class size:
  {max(class_sizes)}

Mean class size:
  {mean_class_size:.2f}

Discernibility Metric:
  {discernibility}

Record accounting:
  PASS

k-anonymity:
  PASS

Output:
  {OUTPUT_FILE}
""".strip() + "\n"

summary_path.write_text(
    summary,
    encoding="utf-8"
)

print()
print(
    f"Saved summary: {summary_path}"
)

print()
print("=" * 70)
print("STANDARD ADULT ARX BASELINE COMPLETE")
print("=" * 70)

########################################################################
# SOURCE NOTEBOOK EXECUTION CELL 27
########################################################################

# Cell 27
# ============================================================
# MILESTONE 3 — ADULT ARX VGH SPECIFICATION
#
# The source comparison specifies:
#   Age              -> height 4, using 5-, 10-, 20-year ranges
#   Gender            -> height 1
#   Race              -> height 1
#   Marital Status    -> height 2
#   Native Country    -> height 2
#   Work Class        -> height 2
#   Occupation        -> height 2
#   Education         -> height 3
#
# Our Adult table uses:
#   sex              instead of gender
#   education.num    instead of education
#
# The exact categorical parent groupings below are an
# operational project decision because the supplied source
# specifies the hierarchy heights but does not provide every
# categorical parent mapping.
# ============================================================

from pathlib import Path
import pandas as pd

ROOT = Path("/content/kc-slice")

DATA = ROOT / "data/processed/adult_standard_baseline_k10_1sa.csv"

VGH_DIR = ROOT / "hierarchies/adult_arx_standard_vgh"
VGH_DIR.mkdir(parents=True, exist_ok=True)

SPEC_FILE = ROOT / "results/adult_vgh_specification.txt"

df = pd.read_csv(DATA)

assert len(df) == 30162

print("=" * 70)
print("MILESTONE 3 — ADULT ARX VGH SPECIFICATION")
print("=" * 70)
print(f"Dataset rows: {len(df):,}")


# ============================================================
# 1. AGE VGH
#
# Level 0: exact age
# Level 1: 5-year interval
# Level 2: 10-year interval
# Level 3: 20-year interval
# Level 4: *
# ============================================================

age_path = VGH_DIR / "age.csv"

with age_path.open("w", encoding="utf-8") as f:

    for age in sorted(df["age"].astype(int).unique()):

        # 5-year bucket
        start5 = (age // 5) * 5
        end5 = start5 + 4
        level1 = f"{start5}-{end5}"

        # 10-year bucket
        start10 = (age // 10) * 10
        end10 = start10 + 9
        level2 = f"{start10}-{end10}"

        # 20-year bucket
        start20 = (age // 20) * 20
        end20 = start20 + 19
        level3 = f"{start20}-{end20}"

        f.write(
            f"{age},{level1},{level2},{level3},*\n"
        )


# ============================================================
# 2. SEX VGH
#
# Source: height 1
# exact -> *
# ============================================================

sex_path = VGH_DIR / "sex.csv"

with sex_path.open("w", encoding="utf-8") as f:

    for value in sorted(
        df["sex"].astype(str).unique()
    ):
        f.write(f"{value},*\n")


# ============================================================
# 3. RACE VGH
#
# Source: height 1
# exact -> *
# ============================================================

race_path = VGH_DIR / "race.csv"

with race_path.open("w", encoding="utf-8") as f:

    for value in sorted(
        df["race"].astype(str).unique()
    ):
        f.write(f"{value},*\n")


# ============================================================
# 4. MARITAL STATUS VGH
#
# Height 2:
#
# exact
#   -> broad marital group
#       -> *
# ============================================================

marital_groups = {
    "Never-married": "Not-married",
    "Divorced": "Not-married",
    "Separated": "Not-married",

    "Married-civ-spouse": "Married",
    "Married-spouse-absent": "Married",
    "Married-AF-spouse": "Married",

    "Widowed": "Widowed",
}

marital_path = VGH_DIR / "marital.status.csv"

for value in df["marital.status"].astype(str).unique():

    assert value in marital_groups, (
        f"Missing marital mapping: {value}"
    )

with marital_path.open("w", encoding="utf-8") as f:

    for value in df["marital.status"].astype(str).unique():

        f.write(
            f"{value},{marital_groups[value]},*\n"
        )


# ============================================================
# 5. WORKCLASS VGH
#
# Height 2:
#
# exact
#   -> employment-sector group
#       -> *
# ============================================================

workclass_groups = {
    "Private": "Private",

    "State-gov": "Government",
    "Federal-gov": "Government",
    "Local-gov": "Government",

    "Self-emp-not-inc": "Self-employed",
    "Self-emp-inc": "Self-employed",

    "Without-pay": "Other",
}

workclass_path = VGH_DIR / "workclass.csv"

for value in df["workclass"].astype(str).unique():

    assert value in workclass_groups, (
        f"Missing workclass mapping: {value}"
    )

with workclass_path.open("w", encoding="utf-8") as f:

    for value in df["workclass"].astype(str).unique():

        f.write(
            f"{value},{workclass_groups[value]},*\n"
        )


# ============================================================
# 6. OCCUPATION VGH
#
# Height 2:
#
# exact
#   -> occupational group
#       -> *
# ============================================================

occupation_groups = {
    "Exec-managerial": "Management",
    "Prof-specialty": "Professional",

    "Adm-clerical": "Office-support",
    "Tech-support": "Office-support",

    "Sales": "Commercial",
    "Other-service": "Service",

    "Protective-serv": "Protective",
    "Handlers-cleaners": "Manual-service",
    "Priv-house-serv": "Manual-service",

    "Craft-repair": "Skilled-trades",
    "Machine-op-inspct": "Skilled-trades",

    "Transport-moving": "Transportation",

    "Farming-fishing": "Agriculture",

    "Armed-Forces": "Military",
}

occupation_path = VGH_DIR / "occupation.csv"

for value in df["occupation"].astype(str).unique():

    assert value in occupation_groups, (
        f"Missing occupation mapping: {value}"
    )

with occupation_path.open("w", encoding="utf-8") as f:

    for value in df["occupation"].astype(str).unique():

        f.write(
            f"{value},{occupation_groups[value]},*\n"
        )


# ============================================================
# 7. NATIVE COUNTRY VGH
#
# Height 2:
#
# exact country
#   -> geographic region
#       -> *
# ============================================================

country_groups = {

    # North America
    "United-States": "North-America",
    "Canada": "North-America",

    # Mexico / Central America / Caribbean
    "Mexico": "Latin-America",
    "Guatemala": "Latin-America",
    "El-Salvador": "Latin-America",
    "Honduras": "Latin-America",
    "Nicaragua": "Latin-America",
    "Panama": "Latin-America",
    "Cuba": "Latin-America",
    "Jamaica": "Latin-America",
    "Puerto-Rico": "Latin-America",
    "Dominican-Republic": "Latin-America",
    "Haiti": "Latin-America",
    "Trinadad&Tobago": "Latin-America",
    "Outlying-US(Guam-USVI-etc)": "Latin-America",

    # South America
    "Columbia": "Latin-America",
    "Ecuador": "Latin-America",
    "Peru": "Latin-America",

    # Europe
    "England": "Europe",
    "Scotland": "Europe",
    "Ireland": "Europe",
    "France": "Europe",
    "Germany": "Europe",
    "Greece": "Europe",
    "Holand-Netherlands": "Europe",
    "Italy": "Europe",
    "Portugal": "Europe",
    "Poland": "Europe",
    "Yugoslavia": "Europe",
    "Hungary": "Europe",

    # Asia
    "China": "Asia",
    "Hong": "Asia",
    "India": "Asia",
    "Iran": "Asia",
    "Japan": "Asia",
    "Cambodia": "Asia",
    "Laos": "Asia",
    "Philippines": "Asia",
    "Taiwan": "Asia",
    "Thailand": "Asia",
    "Vietnam": "Asia",
    "South": "Asia",
}

country_path = VGH_DIR / "native.country.csv"

country_values = df["native.country"].astype(str).unique()

for value in country_values:

    assert value in country_groups, (
        f"Missing native.country mapping: {value}"
    )

with country_path.open("w", encoding="utf-8") as f:

    for value in country_values:

        f.write(
            f"{value},{country_groups[value]},*\n"
        )


# ============================================================
# 8. EDUCATION.NUM VGH
#
# Source Adult Education has 16 distinct values and height 3.
#
# Operational mapping:
#
# exact
#   -> 4-level education band
#       -> broad education group
#           -> *
# ============================================================

education_groups = {
    1:  "Basic-1-4",
    2:  "Basic-1-4",
    3:  "Basic-1-4",
    4:  "Basic-1-4",

    5:  "Basic-5-8",
    6:  "Basic-5-8",
    7:  "Basic-5-8",
    8:  "Basic-5-8",

    9:  "Secondary",
    10: "Secondary",
    11: "Secondary",
    12: "Secondary",

    13: "Post-secondary",
    14: "Post-secondary",
    15: "Post-secondary",
    16: "Post-secondary",
}

education_broad_groups = {
    "Basic-1-4": "Basic",
    "Basic-5-8": "Basic",
    "Secondary": "Secondary",
    "Post-secondary": "Post-secondary",
}

education_path = VGH_DIR / "education.num.csv"

education_values = sorted(
    df["education.num"].astype(int).unique()
)

for value in education_values:

    assert value in education_groups, (
        f"Missing education.num mapping: {value}"
    )

with education_path.open("w", encoding="utf-8") as f:

    for value in education_values:

        level1 = education_groups[value]
        level2 = education_broad_groups[level1]

        f.write(
            f"{value},{level1},{level2},*\n"
        )


# ============================================================
# 9. Verify hierarchy files
# ============================================================

expected_files = [
    "age.csv",
    "sex.csv",
    "race.csv",
    "marital.status.csv",
    "native.country.csv",
    "workclass.csv",
    "occupation.csv",
    "education.num.csv",
]

print()
print("Hierarchy files:")

for filename in expected_files:

    path = VGH_DIR / filename

    assert path.exists(), (
        f"Missing hierarchy: {path}"
    )

    rows = sum(
        1
        for _ in path.open(
            "r",
            encoding="utf-8"
        )
    )

    print(
        f"  {filename:<22} {rows:>3} leaf mappings"
    )


# ============================================================
# 10. Save exact hierarchy specification
# ============================================================

spec = """
MILESTONE 3 — ADULT ARX VGH SPECIFICATION
==========================================

Dataset:
  Adult Census Income

Rows:
  30162

Privacy parameter:
  k = 10

Hierarchy source:
  The comparison paper specifies the Adult VGH attribute
  cardinalities and hierarchy heights.

Source-specified structure:
  age:
      height 4
      exact -> 5-year -> 10-year -> 20-year -> *

  sex/gender:
      height 1
      exact -> *

  race:
      height 1
      exact -> *

  marital.status:
      height 2
      exact -> group -> *

  native.country:
      height 2
      exact -> geographic region -> *

  workclass:
      height 2
      exact -> employment-sector group -> *

  occupation:
      height 2
      exact -> occupational group -> *

  education:
      height 3

Project representation:
  education.num

  exact -> education band -> broad education group -> *

IMPORTANT:
  The source specifies the hierarchy heights and age range
  structure, but does not provide the complete categorical
  parent mapping for every Adult value.

  Therefore the categorical parent mappings used here are
  explicit operational project decisions and will be held
  fixed across the ARX runs rather than changed between
  sensitive-attribute levels.

This hierarchy is intended for the ARX generalization arm.
Mondrian uses partition-based generalization and does not
consume a VGH in the same way.
""".strip() + "\n"

SPEC_FILE.write_text(
    spec,
    encoding="utf-8"
)

print()
print(f"Saved VGH specification: {SPEC_FILE}")

print()
print("=" * 70)
print("ADULT ARX VGH SPECIFICATION COMPLETE")
print("=" * 70)

########################################################################
# SOURCE NOTEBOOK EXECUTION CELL 28
########################################################################

# Cell 28
# ============================================================
# MILESTONE 3 — STANDARD ADULT ARX BASELINE WITH FIXED VGH
# ============================================================

from pathlib import Path
from collections import Counter
import subprocess
import time
import pandas as pd

ROOT = Path("/content/kc-slice")

ARX_DIR = ROOT / "methods/arx"
JAR = ARX_DIR / "libarx-3.9.2.jar"

DATA = ROOT / "data/processed/adult_standard_baseline_k10_1sa.csv"

# IMPORTANT:
# Use the VGH created in Cell 27.
HIER_DIR = ROOT / "hierarchies/adult_arx_standard_vgh"

OUT_DIR = ROOT / "results/arx_standard_adult_vgh"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JAVA_FILE = ARX_DIR / "AdultArxBaseline.java"
OUTPUT_FILE = OUT_DIR / "adult_standard_arx_vgh_k10.csv"

K = 10

QID_COLUMNS = [
    "age",
    "workclass",
    "education.num",
    "marital.status",
    "occupation",
    "race",
    "sex",
    "native.country",
]

SENSITIVE_COLUMN = "income"


# ============================================================
# 1. Verify prerequisites
# ============================================================

assert DATA.exists(), f"Missing input dataset: {DATA}"
assert JAR.exists(), f"Missing ARX JAR: {JAR}"
assert HIER_DIR.exists(), f"Missing VGH directory: {HIER_DIR}"

df = pd.read_csv(DATA)

assert len(df) == 30162

assert list(df.columns) == QID_COLUMNS + [SENSITIVE_COLUMN]

print("=" * 70)
print("MILESTONE 3 — STANDARD ADULT ARX BASELINE WITH FIXED VGH")
print("=" * 70)
print(f"Input rows : {len(df):,}")
print(f"QID count  : {len(QID_COLUMNS)}")
print(f"Sensitive  : {SENSITIVE_COLUMN}")
print(f"k          : {K}")
print(f"ARX JAR    : {JAR}")
print(f"VGH DIR    : {HIER_DIR}")


# ============================================================
# 2. Verify all hierarchy files
# ============================================================

print()
print("Checking VGH files...")

required_hierarchies = [
    "age.csv",
    "workclass.csv",
    "education.num.csv",
    "marital.status.csv",
    "occupation.csv",
    "race.csv",
    "sex.csv",
    "native.country.csv",
]

for filename in required_hierarchies:

    path = HIER_DIR / filename

    assert path.exists(), (
        f"Missing hierarchy file: {path}"
    )

    print(f"  {filename:<22} PASS")


# ============================================================
# 3. Generate Java ARX runner
# ============================================================

print()
print("Generating Java ARX runner...")

java_source = r'''
import java.io.File;
import java.nio.charset.StandardCharsets;

import org.deidentifier.arx.ARXAnonymizer;
import org.deidentifier.arx.ARXConfiguration;
import org.deidentifier.arx.ARXResult;
import org.deidentifier.arx.AttributeType;
import org.deidentifier.arx.Data;
import org.deidentifier.arx.criteria.KAnonymity;
import org.deidentifier.arx.metric.Metric;

public class AdultArxBaseline {

    public static void main(String[] args) throws Exception {

        if (args.length != 3) {
            System.err.println(
                "Usage: AdultArxBaseline <input.csv> <hierarchy_dir> <output.csv>"
            );
            System.exit(1);
        }

        String inputPath = args[0];
        String hierarchyDir = args[1];
        String outputPath = args[2];

        // ----------------------------------------------------
        // Load dataset
        // ----------------------------------------------------

        System.out.println("Loading ARX input...");

        Data data = Data.create(
            inputPath,
            StandardCharsets.UTF_8,
            ','
        );

        // ----------------------------------------------------
        // QIDs + fixed project VGH
        // ----------------------------------------------------

        String[] qids = {
            "age",
            "workclass",
            "education.num",
            "marital.status",
            "occupation",
            "race",
            "sex",
            "native.country"
        };

        for (String qid : qids) {

            File hierarchyFile = new File(
                hierarchyDir,
                qid + ".csv"
            );

            data.getDefinition().setAttributeType(
                qid,
                AttributeType.QUASI_IDENTIFYING_ATTRIBUTE
            );

            data.getDefinition().setHierarchy(
                qid,
                AttributeType.Hierarchy.create(
                    hierarchyFile,
                    StandardCharsets.UTF_8,
                    ','
                )
            );
        }

        // ----------------------------------------------------
        // Income
        //
        // Kept unchanged for this k-anonymity baseline.
        // It is not part of the QID equivalence relation.
        // ----------------------------------------------------

        data.getDefinition().setAttributeType(
            "income",
            AttributeType.INSENSITIVE_ATTRIBUTE
        );

        // ----------------------------------------------------
        // ARX configuration
        // ----------------------------------------------------

        ARXConfiguration config =
            ARXConfiguration.create();

        config.addPrivacyModel(
            new KAnonymity(10)
        );

        // Zero tuple suppression.
        config.setSuppressionLimit(0d);

        // ARX quality metric for optimization.
        config.setQualityModel(
            Metric.createLossMetric()
        );

        // ----------------------------------------------------
        // Run anonymization
        // ----------------------------------------------------

        System.out.println("Running ARX...");

        ARXResult result =
            new ARXAnonymizer().anonymize(
                data,
                config
            );

        // ----------------------------------------------------
        // Verify ARX found a valid optimum
        // ----------------------------------------------------

        if (!result.getOptimumFound()) {

            throw new RuntimeException(
                "ARX did not find a valid optimum."
            );
        }

        if (result.getGlobalOptimum() == null) {

            throw new RuntimeException(
                "ARX returned a null global optimum."
            );
        }

        System.out.println(
            "Global optimum found: PASS"
        );

        // ----------------------------------------------------
        // Save anonymized dataset
        // ----------------------------------------------------

        System.out.println(
            "Saving anonymized output..."
        );

        result
            .getOutput(false)
            .save(
                outputPath,
                ','
            );

        System.out.println(
            "Output saved: " + outputPath
        );
    }
}
'''

JAVA_FILE.write_text(
    java_source,
    encoding="utf-8"
)

print(f"Java source: {JAVA_FILE}")


# ============================================================
# 4. Compile
# ============================================================

print()
print("Compiling ARX runner...")

compile_cmd = [
    "javac",
    "-cp",
    str(JAR),
    str(JAVA_FILE),
]

compile_result = subprocess.run(
    compile_cmd,
    capture_output=True,
    text=True
)

if compile_result.stdout.strip():
    print(compile_result.stdout)

if compile_result.stderr.strip():
    print(compile_result.stderr)

if compile_result.returncode != 0:

    raise RuntimeError(
        "ARX Java compilation failed."
    )

print("Compilation: PASS")


# ============================================================
# 5. Run ARX
# ============================================================

print()
print("Running ARX anonymization...")

run_cmd = [
    "java",
    "-Xmx4g",
    "-cp",
    f"{JAR}:{ARX_DIR}",
    "AdultArxBaseline",
    str(DATA),
    str(HIER_DIR),
    str(OUTPUT_FILE),
]

start_time = time.perf_counter()

run_result = subprocess.run(
    run_cmd,
    capture_output=True,
    text=True
)

runtime_seconds = (
    time.perf_counter() - start_time
)

if run_result.stdout.strip():
    print(run_result.stdout)

if run_result.stderr.strip():
    print("ARX stderr:")
    print(run_result.stderr)

if run_result.returncode != 0:

    raise RuntimeError(
        "ARX anonymization failed."
    )

print(
    f"Python-measured runtime (seconds): "
    f"{runtime_seconds:.4f}"
)


# ============================================================
# 6. Verify output
# ============================================================

assert OUTPUT_FILE.exists(), (
    f"ARX output not found: {OUTPUT_FILE}"
)

arx_output = pd.read_csv(
    OUTPUT_FILE
)

print()
print("=" * 70)
print("ARX OUTPUT VERIFICATION")
print("=" * 70)

print(
    f"Input rows  : {len(df):,}"
)

print(
    f"Output rows : {len(arx_output):,}"
)


# ============================================================
# 7. Record accounting
# ============================================================

assert len(arx_output) == len(df), (
    "Record accounting failed: "
    f"{len(df):,} -> {len(arx_output):,}"
)

print(
    "Record accounting       : PASS"
)


# ============================================================
# 8. Realized equivalence classes
# ============================================================

equivalence_classes = Counter(
    tuple(row[col] for col in QID_COLUMNS)
    for _, row in arx_output.iterrows()
)

class_sizes = list(
    equivalence_classes.values()
)

assert len(class_sizes) > 0

discernibility = sum(
    size ** 2
    for size in class_sizes
)

mean_class_size = (
    sum(class_sizes) /
    len(class_sizes)
)

print()
print(
    f"Equivalence classes    : "
    f"{len(class_sizes):,}"
)

print(
    f"Minimum class size     : "
    f"{min(class_sizes):,}"
)

print(
    f"Maximum class size     : "
    f"{max(class_sizes):,}"
)

print(
    f"Mean class size        : "
    f"{mean_class_size:.2f}"
)

print(
    f"Discernibility Metric  : "
    f"{discernibility:,}"
)


# ============================================================
# 9. Verify k-anonymity
# ============================================================

assert min(class_sizes) >= K, (
    "k-anonymity verification failed: "
    f"minimum class size = {min(class_sizes)}"
)

assert sum(class_sizes) == len(arx_output)

print(
    "k=10 anonymity          : PASS"
)


# ============================================================
# 10. Save final ARX VGH baseline summary
# ============================================================

summary_path = (
    ROOT /
    "results" /
    "milestone3_arx_vgh_baseline.txt"
)

summary = f"""
MILESTONE 3 — STANDARD ADULT ARX BASELINE WITH FIXED VGH

Dataset:
  Adult Census Income

Input records:
  {len(df)}

Output records:
  {len(arx_output)}

QIDs:
{chr(10).join("  - " + q for q in QID_COLUMNS)}

Non-QID attribute:
  {SENSITIVE_COLUMN}

Privacy model:
  k-anonymity

k:
  {K}

Suppression limit:
  0

ARX version:
  3.9.2

Hierarchy:
  {HIER_DIR}

Runtime (seconds):
  {runtime_seconds:.4f}

Equivalence classes:
  {len(class_sizes)}

Minimum class size:
  {min(class_sizes)}

Maximum class size:
  {max(class_sizes)}

Mean class size:
  {mean_class_size:.2f}

Discernibility Metric:
  {discernibility}

Record accounting:
  PASS

k-anonymity:
  PASS

Output:
  {OUTPUT_FILE}
""".strip() + "\n"

summary_path.write_text(
    summary,
    encoding="utf-8"
)

print()
print(
    f"Saved summary: {summary_path}"
)

print()
print("=" * 70)
print("STANDARD ADULT ARX VGH BASELINE COMPLETE")
print("=" * 70)

########################################################################
# SOURCE NOTEBOOK EXECUTION CELL 29
########################################################################

# Cell 29
# ============================================================
# ARX BASELINE — TRANSFORMATION AUDIT
# Inspect how much each QID was generalized.
# ============================================================

from pathlib import Path
import pandas as pd

ROOT = Path("/content/kc-slice")

OUTPUT_FILE = (
    ROOT /
    "results/arx_standard_adult_vgh" /
    "adult_standard_arx_vgh_k10.csv"
)

QID_COLUMNS = [
    "age",
    "workclass",
    "education.num",
    "marital.status",
    "occupation",
    "race",
    "sex",
    "native.country",
]

df_arx = pd.read_csv(OUTPUT_FILE)

print("=" * 70)
print("ARX TRANSFORMATION AUDIT")
print("=" * 70)

print(f"Rows: {len(df_arx):,}")
print()

for col in QID_COLUMNS:

    values = df_arx[col].dropna().astype(str)

    print("-" * 70)
    print(f"{col}")
    print(f"Unique generalized values: {values.nunique():,}")

    sample_values = values.drop_duplicates().tolist()[:20]

    print("Sample values:")
    for value in sample_values:
        print(f"  {value}")

print()
print("=" * 70)
print("TRANSFORMATION AUDIT COMPLETE")
print("=" * 70)

########################################################################
# SOURCE NOTEBOOK EXECUTION CELL 42
########################################################################

# Cell 42 — MILESTONE 3
# STANDARD ADULT MONDRIAN — ROW-LEVEL OUTPUT
#
# Recreates the already-verified strict 8-QID baseline.
#
# Important:
#   First 8 columns = anonymized QIDs
#   9th column      = income (preserved)
#   10th column     = _source_row (preserved)
#
# QI_num=8 tells the upstream implementation to anonymize
# only the first 8 columns.
# ============================================================

from pathlib import Path
from collections import Counter
import copy
import sys

import pandas as pd

ROOT = Path("/content/kc-slice")

MONDRIAN_PATH = (
    ROOT /
    "methods/mondrian/upstream"
)

if str(MONDRIAN_PATH) not in sys.path:
    sys.path.insert(
        0,
        str(MONDRIAN_PATH)
    )

from mondrian import mondrian


# ============================================================
# Configuration
# ============================================================

K = 10
RELAX = False
QI_NUM = 8

QID_COLUMNS = [
    "age",
    "workclass",
    "education.num",
    "marital.status",
    "occupation",
    "race",
    "sex",
    "native.country",
]

TARGET = "income"

SOURCE_ID = "_source_row"


print("=" * 70)
print("STANDARD ADULT MONDRIAN — ROW-LEVEL OUTPUT")
print("=" * 70)

print(
    f"Records     : 30,162"
)

print(
    f"QID count   : {QI_NUM}"
)

print(
    f"k           : {K}"
)

print(
    f"Model       : strict"
)


# ============================================================
# Load the exact baseline used for the verified run
# ============================================================

baseline_path = (
    ROOT /
    "data/processed/"
    "adult_standard_baseline_k10_1sa.csv"
)

df = pd.read_csv(
    baseline_path
)

assert len(df) == 30162

for column in QID_COLUMNS + [TARGET]:
    assert column in df.columns


print()
print(
    "Baseline loaded            : PASS"
)


# ============================================================
# Preserve original row identity
#
# The baseline retains the original Adult dataframe index
# after '?' rows were removed. Reset to a clean 0..N-1
# evaluation index so it exactly matches the fixed split
# generated in Cell 39.
# ============================================================

df = df.reset_index(
    drop=True
)

df[SOURCE_ID] = df.index


# ============================================================
# Encode categorical QIDs
#
# The original qiyuangong/Mondrian implementation uses an
# intuitive ordering for categorical attributes rather than
# generalization hierarchies. We retain the project's
# first-appearance ordering used in the verified baseline.
# ============================================================

categorical_qids = [
    column
    for column in QID_COLUMNS
    if column not in {
        "age",
        "education.num",
    }
]

category_orders = {}
category_maps = {}

print()
print("Categorical ordering:")

for column in categorical_qids:

    order = list(
        dict.fromkeys(
            df[column]
            .astype(str)
            .tolist()
        )
    )

    mapping = {
        value: index
        for index, value
        in enumerate(order)
    }

    category_orders[column] = order
    category_maps[column] = mapping

    print(
        f"  {column}: {order}"
    )


# ============================================================
# Build records
#
# IMPORTANT:
#   [8 QIDs] + [income] + [_source_row]
#
# QI_num=8 means Mondrian anonymizes only the first 8.
# income and _source_row survive untouched.
# ============================================================

data = []

for _, row in df.iterrows():

    record = []

    for column in QID_COLUMNS:

        if column in category_maps:

            record.append(
                category_maps[column][
                    str(row[column])
                ]
            )

        else:

            record.append(
                int(row[column])
            )

    # Target remains untouched.
    record.append(
        str(row[TARGET])
    )

    # Internal evaluation identifier remains untouched.
    record.append(
        int(row[SOURCE_ID])
    )

    data.append(record)


assert len(data) == 30162
assert all(
    len(record) == 10
    for record in data
)

print()
print(
    "Mondrian input construction : PASS"
)


# ============================================================
# Run strict Mondrian
# ============================================================

print()
print(
    "Running strict Mondrian..."
)

result, eval_result = mondrian(
    copy.deepcopy(data),
    K,
    RELAX,
    QI_num=QI_NUM
)

ncp = float(
    eval_result[0]
)

runtime = float(
    eval_result[1]
)


# ============================================================
# Calculate equivalence classes and DM
# ============================================================

equivalence_classes = Counter(
    tuple(
        record[:QI_NUM]
    )
    for record in result
)

class_sizes = list(
    equivalence_classes.values()
)

assert len(result) == 30162

assert len(class_sizes) > 0

discernibility = sum(
    size ** 2
    for size in class_sizes
)


# ============================================================
# Verify target + source IDs survived unchanged
# ============================================================

source_ids = [
    int(record[9])
    for record in result
]

income_values = [
    str(record[8])
    for record in result
]

assert set(source_ids) == set(
    range(30162)
)

assert len(source_ids) == len(
    set(source_ids)
)

print(
    "Source-row preservation      : PASS"
)


original_income = (
    df[TARGET]
    .astype(str)
    .value_counts()
    .sort_index()
)

result_income = (
    pd.Series(income_values)
    .value_counts()
    .sort_index()
)

assert original_income.equals(
    result_income
)

print(
    "Income-target preservation  : PASS"
)


# ============================================================
# Build row-level output
#
# QID generalized values are kept exactly as produced by
# Mondrian.
#
# Categorical outputs may therefore be encoded generalized
# intervals because this upstream implementation works on
# ordered encoded categorical values.
# ============================================================

output_records = []

for record in result:

    output_records.append([
        record[i]
        for i in range(QI_NUM)
    ] + [
        record[8],
        record[9],
    ])


mondrian_output = pd.DataFrame(
    output_records,
    columns=QID_COLUMNS + [
        TARGET,
        SOURCE_ID,
    ]
)


# ============================================================
# Verification
# ============================================================

print()
print("=" * 70)
print("MONDRIAN RESULTS")
print("=" * 70)

print(
    f"Input records          : {len(df):,}"
)

print(
    f"Output records         : {len(mondrian_output):,}"
)

print(
    f"NCP (%)                : {ncp:.4f}"
)

print(
    f"Runtime (seconds)      : {runtime:.4f}"
)

print(
    f"Equivalence classes    : {len(class_sizes):,}"
)

print(
    f"Minimum class size     : {min(class_sizes):,}"
)

print(
    f"Maximum class size     : {max(class_sizes):,}"
)

print(
    f"Mean class size        : "
    f"{sum(class_sizes) / len(class_sizes):.2f}"
)

print(
    f"Discernibility Metric  : "
    f"{discernibility:,}"
)


# ============================================================
# K-anonymity
# ============================================================

assert (
    min(class_sizes)
    >=
    K
)

print()
print(
    "k=10 anonymity          : PASS"
)


# ============================================================
# Complete record accounting
# ============================================================

assert len(mondrian_output) == 30162

assert (
    mondrian_output[
        SOURCE_ID
    ]
    .nunique()
    ==
    30162
)

print(
    "Record accounting       : PASS"
)


# ============================================================
# Save row-level output
# ============================================================

RESULTS_DIR = (
    ROOT /
    "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ROW_OUTPUT = (
    RESULTS_DIR /
    "adult_standard_mondrian_k10_1sa.csv"
)

mondrian_output.to_csv(
    ROW_OUTPUT,
    index=False
)


# ============================================================
# Save summary
# ============================================================

SUMMARY_OUTPUT = (
    RESULTS_DIR /
    "milestone3_mondrian_standard_8qid_rowlevel.txt"
)

summary = f"""
MILESTONE 3 — STANDARD ADULT MONDRIAN ROW-LEVEL OUTPUT

Dataset: Adult Census Income
Input records: {len(df)}
Output records: {len(mondrian_output)}

QIDs: {", ".join(QID_COLUMNS)}
Sensitive attribute: income

k: {K}
Model: strict
QI_num: {QI_NUM}

NCP (%): {ncp:.4f}
Runtime (seconds): {runtime:.4f}

Equivalence classes: {len(class_sizes)}
Minimum class size: {min(class_sizes)}
Maximum class size: {max(class_sizes)}
Mean class size: {sum(class_sizes)/len(class_sizes):.2f}

Discernibility Metric: {discernibility}

Record accounting: PASS
Source-row preservation: PASS
Income preservation: PASS
k-anonymity verification: PASS

Row-level output:
{ROW_OUTPUT}
""".strip() + "\n"

SUMMARY_OUTPUT.write_text(
    summary,
    encoding="utf-8"
)


# ============================================================
# Display sample
# ============================================================

print()
print(
    "Row-level output sample:"
)

print(
    mondrian_output.head(
        10
    ).to_string(
        index=False
    )
)

print()
print(
    f"Saved row-level output : {ROW_OUTPUT}"
)

print(
    f"Saved summary          : {SUMMARY_OUTPUT}"
)

print()
print("=" * 70)
print("STANDARD ADULT MONDRIAN ROW-LEVEL OUTPUT COMPLETE")
print("=" * 70)

########################################################################
# SOURCE NOTEBOOK EXECUTION CELL 47
########################################################################

# Cell 46 — MILESTONE 3
# CREATE CANONICAL ADULT VGH SPECIFICATION — FIXED
# ============================================================

from pathlib import Path
import pandas as pd
import yaml

ROOT = Path("/content/kc-slice")

VGH_DIR = (
    ROOT /
    "hierarchies/adult_arx_standard_vgh"
)

OUTPUT = (
    ROOT /
    "hierarchies/adult_hierarchies.yaml"
)

EXPECTED_ATTRIBUTES = [
    "age",
    "workclass",
    "education.num",
    "marital.status",
    "occupation",
    "race",
    "sex",
    "native.country",
]

# ============================================================
# IMPORTANT:
# The existing hierarchy CSV files are HEADERLESS.
#
# Therefore:
#   header=None
#
# is required.
#
# The first column is always the leaf value.
# Remaining columns are hierarchy/generalization levels.
# ============================================================

LEVELS_BY_ATTRIBUTE = {
    "age": [
        "15-19",
        "10-19",
        "0-19",
        "*",
    ],

    "workclass": [
        "Private.1",
        "*",
    ],

    "education.num": [
        "Basic-1-4",
        "Basic",
        "*",
    ],

    "marital.status": [
        "Widowed.1",
        "*",
    ],

    "occupation": [
        "Management",
        "*",
    ],

    "race": [
        "*",
    ],

    "sex": [
        "*",
    ],

    "native.country": [
        "North-America",
        "*",
    ],
}

EXPECTED_LEAF_COUNTS = {
    "age": 72,
    "workclass": 7,
    "education.num": 16,
    "marital.status": 7,
    "occupation": 14,
    "race": 5,
    "sex": 2,
    "native.country": 41,
}


print("=" * 70)
print("MILESTONE 3 — CANONICAL ADULT VGH")
print("=" * 70)


# ============================================================
# Verify source directory
# ============================================================

assert VGH_DIR.exists(), (
    f"VGH directory not found: {VGH_DIR}"
)

print()
print(
    "VGH directory exists : PASS"
)


# ============================================================
# Build canonical hierarchy representation
# ============================================================

hierarchies = {}

for attribute in EXPECTED_ATTRIBUTES:

    path = (
        VGH_DIR /
        f"{attribute}.csv"
    )

    assert path.exists(), (
        f"Missing hierarchy file: {path}"
    )

    # --------------------------------------------------------
    # CRITICAL FIX:
    # These files do NOT contain a header row.
    # --------------------------------------------------------

    table = pd.read_csv(
        path,
        header=None,
        dtype=str,
    )

    table = table.dropna(
        how="all"
    )

    table = table.reset_index(
        drop=True
    )

    expected_leaf_count = (
        EXPECTED_LEAF_COUNTS[
            attribute
        ]
    )

    expected_levels = (
        LEVELS_BY_ATTRIBUTE[
            attribute
        ]
    )

    expected_columns = (
        1 +
        len(expected_levels)
    )

    # --------------------------------------------------------
    # Validate dimensions.
    # --------------------------------------------------------

    assert len(table) == expected_leaf_count, (
        f"{attribute}: expected "
        f"{expected_leaf_count} rows, got "
        f"{len(table)}"
    )

    assert table.shape[1] == expected_columns, (
        f"{attribute}: expected "
        f"{expected_columns} columns, got "
        f"{table.shape[1]}"
    )

    # --------------------------------------------------------
    # Convert positional columns into the canonical hierarchy.
    # --------------------------------------------------------

    mappings = []

    for _, row in table.iterrows():

        leaf = str(
            row.iloc[0]
        )

        generalization = {}

        for level_index, level_name in enumerate(
            expected_levels,
            start=1
        ):

            value = row.iloc[
                level_index
            ]

            if pd.isna(value):

                value = None

            else:

                value = str(value)

            generalization[
                level_name
            ] = value

        mappings.append({
            "value": leaf,
            "generalization": generalization,
        })

    hierarchies[attribute] = {
        "levels": expected_levels,
        "mappings": mappings,
    }


# ============================================================
# Add project metadata
# ============================================================

specification = {

    "dataset":
        "Adult Census Income",

    "records_after_cleaning":
        30162,

    "project_parameters": {

        "k":
            10,

        "sensitive_attribute":
            "income",

        "qids":
            EXPECTED_ATTRIBUTES,
    },

    "hierarchy_source":
        "Operational VGH constructed for this project "
        "and used for the Adult baseline.",

    "hierarchy_notes":
        "Categorical parent mappings are explicit project "
        "operationalizations. The source material specifies "
        "hierarchy structure/heights but does not provide "
        "every categorical parent mapping.",

    "attributes":
        hierarchies,
}


# ============================================================
# Write canonical YAML
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    yaml.safe_dump(
        specification,
        f,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )


# ============================================================
# Reload YAML
# ============================================================

with open(
    OUTPUT,
    "r",
    encoding="utf-8"
) as f:

    loaded = yaml.safe_load(f)


# ============================================================
# Top-level verification
# ============================================================

assert (
    loaded["dataset"]
    ==
    "Adult Census Income"
)

assert (
    loaded["records_after_cleaning"]
    ==
    30162
)

assert (
    loaded["project_parameters"]["k"]
    ==
    10
)

assert (
    loaded["project_parameters"][
        "sensitive_attribute"
    ]
    ==
    "income"
)

assert (
    loaded["project_parameters"]["qids"]
    ==
    EXPECTED_ATTRIBUTES
)


# ============================================================
# Verify EVERY hierarchy retained ALL leaf mappings
# ============================================================

print()
print("-" * 70)
print("HIERARCHY VERIFICATION")
print("-" * 70)

for attribute in EXPECTED_ATTRIBUTES:

    data = loaded[
        "attributes"
    ][attribute]

    expected_count = (
        EXPECTED_LEAF_COUNTS[
            attribute
        ]
    )

    actual_count = len(
        data["mappings"]
    )

    assert (
        actual_count
        ==
        expected_count
    )

    print(
        f"{attribute:<20}: "
        f"{actual_count:>2} leaf mappings : PASS"
    )


# ============================================================
# Print hierarchy-level information
# ============================================================

print()
print("-" * 70)
print("CANONICAL HIERARCHY SUMMARY")
print("-" * 70)

for attribute in EXPECTED_ATTRIBUTES:

    data = loaded[
        "attributes"
    ][attribute]

    print()
    print(
        f"[{attribute}]"
    )

    print(
        f"  Leaf mappings : "
        f"{len(data['mappings'])}"
    )

    print(
        f"  Levels        : "
        f"{data['levels']}"
    )


# ============================================================
# Spot-check important mappings
# ============================================================

print()
print("-" * 70)
print("SPOT CHECKS")
print("-" * 70)


# ------------------------------------------------------------
# Age
# ------------------------------------------------------------

age_mappings = {
    item["value"]: item["generalization"]
    for item in loaded[
        "attributes"
    ]["age"]["mappings"]
}

assert age_mappings["17"]["15-19"] == "15-19"
assert age_mappings["17"]["10-19"] == "10-19"
assert age_mappings["17"]["0-19"] == "0-19"
assert age_mappings["17"]["*"] == "*"

print(
    "Age hierarchy mapping     : PASS"
)


# ------------------------------------------------------------
# Workclass
# ------------------------------------------------------------

workclass_mappings = {
    item["value"]: item["generalization"]
    for item in loaded[
        "attributes"
    ]["workclass"]["mappings"]
}

assert (
    workclass_mappings["Private"]["Private.1"]
    ==
    "Private"
)

assert (
    workclass_mappings["State-gov"]["Private.1"]
    ==
    "Government"
)

print(
    "Workclass hierarchy       : PASS"
)


# ------------------------------------------------------------
# Education
# ------------------------------------------------------------

education_mappings = {
    item["value"]: item["generalization"]
    for item in loaded[
        "attributes"
    ]["education.num"]["mappings"]
}

assert (
    education_mappings["1"]["Basic-1-4"]
    ==
    "Basic-1-4"
)

assert (
    education_mappings["13"]["Basic-1-4"]
    ==
    "Post-secondary"
)

print(
    "Education hierarchy       : PASS"
)


# ------------------------------------------------------------
# Occupation
# ------------------------------------------------------------

occupation_mappings = {
    item["value"]: item["generalization"]
    for item in loaded[
        "attributes"
    ]["occupation"]["mappings"]
}

assert (
    occupation_mappings["Exec-managerial"]["Management"]
    ==
    "Management"
)

assert (
    occupation_mappings["Machine-op-inspct"]["Management"]
    ==
    "Skilled-trades"
)

print(
    "Occupation hierarchy      : PASS"
)


# ------------------------------------------------------------
# Native country
# ------------------------------------------------------------

country_mappings = {
    item["value"]: item["generalization"]
    for item in loaded[
        "attributes"
    ]["native.country"]["mappings"]
}

assert (
    country_mappings["United-States"]["North-America"]
    ==
    "North-America"
)

assert (
    country_mappings["South"]["North-America"]
    ==
    "Asia"
)

print(
    "Native-country hierarchy  : PASS"
)


# ============================================================
# Save audit
# ============================================================

AUDIT_OUTPUT = (
    ROOT /
    "results/adult_canonical_vgh_verified.txt"
)

AUDIT_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    AUDIT_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "MILESTONE 3 — CANONICAL ADULT VGH VERIFIED\n"
    )

    f.write("=" * 65 + "\n\n")

    f.write(
        f"Dataset: Adult Census Income\n"
    )

    f.write(
        f"Records after cleaning: 30162\n"
    )

    f.write(
        f"k: 10\n"
    )

    f.write(
        f"Sensitive attribute: income\n\n"
    )

    for attribute in EXPECTED_ATTRIBUTES:

        data = loaded[
            "attributes"
        ][attribute]

        f.write(
            f"{attribute}:\n"
        )

        f.write(
            f"  Leaf mappings: "
            f"{len(data['mappings'])}\n"
        )

        f.write(
            f"  Levels: "
            f"{data['levels']}\n\n"
        )

    f.write(
        "All source hierarchy CSVs were read as "
        "headerless files (header=None).\n"
    )

    f.write(
        "All expected leaf mappings were preserved.\n"
    )


# ============================================================
# FINAL
# ============================================================

print()
print(
    f"Canonical YAML: {OUTPUT}"
)

print(
    f"Verification log: {AUDIT_OUTPUT}"
)

print()
print("=" * 70)
print("CANONICAL ADULT VGH VERIFIED SUCCESSFULLY")
print("=" * 70)

########################################################################
# SOURCE NOTEBOOK EXECUTION CELL 49
########################################################################

# Cell 48 — MILESTONE 3
# FINAL BASELINE RESULTS TABLE + REPRODUCTION NOTE
# ============================================================

from pathlib import Path
from collections import Counter
import pandas as pd

ROOT = Path("/content/kc-slice")

RESULTS_DIR = ROOT / "results"

COMPARISON_DIR = (
    RESULTS_DIR /
    "milestone3_adult_1sa_comparison"
)

COMPARISON_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Paths
# ============================================================

MONDRIAN_PATH = (
    COMPARISON_DIR /
    "adult_standard_mondrian_vgh_k10_1sa.csv"
)

ARX_PATH = (
    RESULTS_DIR /
    "arx_standard_adult_vgh" /
    "adult_standard_arx_vgh_k10.csv"
)

KC_PATH = (
    ROOT /
    "results/kc_slice_adult_1sa" /
    "adult_1sa_evaluation_dataset.csv"
)

BASELINE_PATH = (
    ROOT /
    "data/processed" /
    "adult_standard_baseline_k10_1sa.csv"
)


# ============================================================
# Experiment configuration
# ============================================================

K = 10

SENSITIVE_ATTRIBUTE = "income"

QIDS = [
    "age",
    "workclass",
    "education.num",
    "marital.status",
    "occupation",
    "race",
    "sex",
    "native.country",
]

KC_SLICE_COLUMNS = [
    "slice_1",
    "slice_2",
    "slice_3",
    "slice_4",
    "slice_5",
]


print("=" * 70)
print("MILESTONE 3 — FINAL BASELINE RESULTS")
print("=" * 70)


# ============================================================
# Load final artifacts
# ============================================================

baseline = pd.read_csv(
    BASELINE_PATH
)

mondrian = pd.read_csv(
    MONDRIAN_PATH
)

arx = pd.read_csv(
    ARX_PATH
)

kc = pd.read_csv(
    KC_PATH
)

assert len(baseline) == 30162
assert len(mondrian) == 30162
assert len(arx) == 30162
assert len(kc) == 30162

print()
print(
    "All baseline artifacts loaded : PASS"
)


# ============================================================
# Helper
# ============================================================

def calculate_dm(frame, columns):
    """
    Standard discernibility metric:

        DM = sum(|E|^2)

    where E is each equivalence class.
    """

    class_sizes = (
        frame
        .groupby(
            columns,
            dropna=False
        )
        .size()
    )

    dm = int(
        (class_sizes ** 2).sum()
    )

    return {
        "dm": dm,
        "equivalence_classes": len(
            class_sizes
        ),
        "minimum_class_size": int(
            class_sizes.min()
        ),
        "maximum_class_size": int(
            class_sizes.max()
        ),
        "mean_class_size": float(
            class_sizes.mean()
        ),
    }


# ============================================================
# 1. MONDRIAN
# ============================================================

print()
print("-" * 70)
print("MONDRIAN")
print("-" * 70)

assert all(
    column in mondrian.columns
    for column in QIDS
)

mondrian_stats = calculate_dm(
    mondrian,
    QIDS
)

assert (
    mondrian_stats["minimum_class_size"]
    >=
    K
)

print(
    f"Equivalence classes : "
    f"{mondrian_stats['equivalence_classes']:,}"
)

print(
    f"Minimum class       : "
    f"{mondrian_stats['minimum_class_size']:,}"
)

print(
    f"Maximum class       : "
    f"{mondrian_stats['maximum_class_size']:,}"
)

print(
    f"Mean class          : "
    f"{mondrian_stats['mean_class_size']:.2f}"
)

print(
    f"DM                  : "
    f"{mondrian_stats['dm']:,}"
)

print(
    "k=10 verification   : PASS"
)


# ============================================================
# 2. ARX
# ============================================================

print()
print("-" * 70)
print("ARX")
print("-" * 70)

assert all(
    column in arx.columns
    for column in QIDS
)

arx_stats = calculate_dm(
    arx,
    QIDS
)

assert (
    arx_stats["minimum_class_size"]
    >=
    K
)

print(
    f"Equivalence classes : "
    f"{arx_stats['equivalence_classes']:,}"
)

print(
    f"Minimum class       : "
    f"{arx_stats['minimum_class_size']:,}"
)

print(
    f"Maximum class       : "
    f"{arx_stats['maximum_class_size']:,}"
)

print(
    f"Mean class          : "
    f"{arx_stats['mean_class_size']:.2f}"
)

print(
    f"DM                  : "
    f"{arx_stats['dm']:,}"
)

print(
    "k=10 verification   : PASS"
)


# ============================================================
# 3. KC-SLICE
#
# IMPORTANT:
#
# KC-Slice does NOT use ordinary k-anonymity equivalence
# classes as its privacy mechanism.
#
# Nevertheless, the guide explicitly requests a DM for the
# three methods.
#
# Therefore we calculate a clearly labeled diagnostic:
#
#   DM_sliced = sum(|E_sliced|^2)
#
# where E_sliced is defined by the complete released
# concatenated QID-slice tuple.
#
# This is NOT a claim that KC-Slice provides k-anonymity.
# ============================================================

print()
print("-" * 70)
print("KC-SLICE")
print("-" * 70)

assert all(
    column in kc.columns
    for column in KC_SLICE_COLUMNS
)

kc_stats = calculate_dm(
    kc,
    KC_SLICE_COLUMNS
)

print(
    f"Sliced-QID classes  : "
    f"{kc_stats['equivalence_classes']:,}"
)

print(
    f"Minimum class       : "
    f"{kc_stats['minimum_class_size']:,}"
)

print(
    f"Maximum class       : "
    f"{kc_stats['maximum_class_size']:,}"
)

print(
    f"Mean class          : "
    f"{kc_stats['mean_class_size']:.2f}"
)

print(
    f"Adapted DM          : "
    f"{kc_stats['dm']:,}"
)

print()
print(
    "KC-Slice privacy model: bucket/HSA threshold + slicing"
)

print(
    "KC-Slice k-anonymity claim: NOT made"
)


# ============================================================
# BUILD FINAL BASELINE TABLE
# ============================================================

baseline_results = pd.DataFrame([

    {
        "method":
            "Mondrian",

        "k":
            K,

        "sensitive_attribute":
            SENSITIVE_ATTRIBUTE,

        "qid_count":
            len(QIDS),

        "discernibility_metric":
            mondrian_stats["dm"],

        "equivalence_classes":
            mondrian_stats["equivalence_classes"],

        "minimum_class_size":
            mondrian_stats["minimum_class_size"],

        "maximum_class_size":
            mondrian_stats["maximum_class_size"],

        "mean_class_size":
            mondrian_stats["mean_class_size"],

        "dm_definition":
            "sum(|E|^2) over generalized QID equivalence classes",

        "privacy_note":
            "Strict k-anonymity",
    },

    {
        "method":
            "ARX",

        "k":
            K,

        "sensitive_attribute":
            SENSITIVE_ATTRIBUTE,

        "qid_count":
            len(QIDS),

        "discernibility_metric":
            arx_stats["dm"],

        "equivalence_classes":
            arx_stats["equivalence_classes"],

        "minimum_class_size":
            arx_stats["minimum_class_size"],

        "maximum_class_size":
            arx_stats["maximum_class_size"],

        "mean_class_size":
            arx_stats["mean_class_size"],

        "dm_definition":
            "sum(|E|^2) over generalized QID equivalence classes",

        "privacy_note":
            "k-anonymity with explicit Adult VGH",
    },

    {
        "method":
            "KC-Slice",

        "k":
            K,

        "sensitive_attribute":
            SENSITIVE_ATTRIBUTE,

        "qid_count":
            len(KC_SLICE_COLUMNS),

        "discernibility_metric":
            kc_stats["dm"],

        "equivalence_classes":
            kc_stats["equivalence_classes"],

        "minimum_class_size":
            kc_stats["minimum_class_size"],

        "maximum_class_size":
            kc_stats["maximum_class_size"],

        "mean_class_size":
            kc_stats["mean_class_size"],

        "dm_definition":
            "Adapted DM over complete released sliced-QID tuples",

        "privacy_note":
            "KC-Slice bucket/HSA privacy model; not k-anonymity",
    },
])


# ============================================================
# Save required guide-style table
# ============================================================

BASELINE_OUTPUT = (
    RESULTS_DIR /
    "milestone3_baseline.csv"
)

baseline_results.to_csv(
    BASELINE_OUTPUT,
    index=False
)


print()
print("=" * 70)
print("FINAL BASELINE RESULTS TABLE")
print("=" * 70)

print(
    baseline_results[
        [
            "method",
            "k",
            "sensitive_attribute",
            "discernibility_metric",
            "equivalence_classes",
            "minimum_class_size",
            "maximum_class_size",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Published-reference/reproduction note
# ============================================================

NOTE_OUTPUT = (
    RESULTS_DIR /
    "milestone3_reproduction_note.txt"
)

note = f"""
MILESTONE 3 — REPRODUCTION NOTE
================================

Dataset
-------
Adult Census Income

Raw records: 32,561
Records after removing rows containing '?': 30,162

Quasi-identifiers
-----------------
{", ".join(QIDS)}

Sensitive attribute
-------------------
{SENSITIVE_ATTRIBUTE}

Privacy parameter
-----------------
k = {K}


Hierarchy specification
-----------------------
A canonical Adult hierarchy specification was created at:

  hierarchies/adult_hierarchies.yaml

The hierarchy contains explicit mappings for:
  age
  workclass
  education.num
  marital.status
  occupation
  race
  sex
  native.country

Mondrian
--------
Strict qiyuangong/Mondrian was used for partitioning.

The final released QID values were projected onto the same
hand-built Adult VGH used for the ARX baseline.

Equivalence classes after VGH projection:
  {mondrian_stats["equivalence_classes"]}

Minimum class size:
  {mondrian_stats["minimum_class_size"]}

Maximum class size:
  {mondrian_stats["maximum_class_size"]}

Discernibility Metric:
  {mondrian_stats["dm"]:,}


ARX
---
ARX 3.9.2 was used with the explicit Adult VGH.

Equivalence classes:
  {arx_stats["equivalence_classes"]}

Minimum class size:
  {arx_stats["minimum_class_size"]}

Maximum class size:
  {arx_stats["maximum_class_size"]}

Discernibility Metric:
  {arx_stats["dm"]:,}


KC-Slice
--------
KC-Slice was implemented from the source algorithm.

For the 1-SA Adult baseline:
  Sensitive attribute = income
  HSA = <=50K
  C = 5%
  Internal bucket size = 15,081

HSA suppression:
  21,146 records
  70.1081% of the dataset

Because KC-Slice does not use ordinary k-anonymity as its
privacy model, its DM is reported only as an adapted diagnostic
over the complete released sliced-QID tuple.

Adapted sliced-QID DM:
  {kc_stats["dm"]:,}

This number must not be interpreted as proof of k-anonymity.


Discernibility Metric definition
--------------------------------
For Mondrian and ARX:

  DM = sum(|E|^2)

where E denotes a generalized-QID equivalence class.


Published-reference check
-------------------------
LeFevre, DeWitt and Ramakrishnan, "Mondrian Multidimensional
K-Anonymity" includes an Adult-dataset comparison using the
Discernibility Penalty in Figure 10.

The source presents the Adult comparison graphically rather than
as a table of exact numerical values in the accompanying text.
Therefore an exact numerical equality/ballpark claim is not made
here without extracting a precise point from the original figure.

The published comparison also uses its own experimental
configuration, so differences in QID selection, hierarchy
definition, dataset preprocessing, and implementation details
must be considered when comparing numerical results.


Method gap
----------
The Mondrian and ARX DM values can differ substantially even
with k=10 because their anonymization/search procedures and
resulting equivalence-class structures are not identical.

In this experiment the hierarchy specification is controlled,
but the upstream Mondrian implementation does not directly
consume the external VGH during partition selection. Its final
partitions are therefore expressed through the shared VGH before
the released DM is calculated.

KC-Slice is not directly comparable to Mondrian/ARX under the
same DM interpretation because its privacy mechanism is based on
bucket formation, HSA frequency thresholds, suppression, SIDs,
and slicing rather than ordinary k-anonymity.


Final baseline table
--------------------
{baseline_results[
    [
        "method",
        "k",
        "sensitive_attribute",
        "discernibility_metric",
    ]
].to_string(index=False)}


Milestone status
----------------
Adult 1-SA baseline reproduction and the requested
three-method baseline results table have been generated.
""".strip() + "\n"


NOTE_OUTPUT.write_text(
    note,
    encoding="utf-8"
)


# ============================================================
# Verification
# ============================================================

assert (
    set(
        baseline_results["method"]
    )
    ==
    {
        "Mondrian",
        "ARX",
        "KC-Slice",
    }
)

assert len(
    baseline_results
) == 3

assert all(
    baseline_results[
        "discernibility_metric"
    ]
    > 0
)

assert BASELINE_OUTPUT.exists()
assert NOTE_OUTPUT.exists()


print()
print("=" * 70)
print("SUBMISSION ARTIFACTS CREATED")
print("=" * 70)

print(
    f"Baseline table     : {BASELINE_OUTPUT}"
)

print(
    f"Reproduction note  : {NOTE_OUTPUT}"
)

print()
print("=" * 70)
print("MILESTONE 3 BASELINE RESULTS COMPLETE")
print("=" * 70)
