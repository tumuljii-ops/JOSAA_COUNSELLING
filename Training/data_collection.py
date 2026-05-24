import pandas as pd


# =====================================================
# Function:
# Convert institute name -> college type
# =====================================================

def get_college_type(name):

    # IIT colleges
    if "Indian Institute of Technology" in name:
        return "IIT"

    # NIT colleges
    elif "National Institute of Technology" in name:
        return "NIT"

    # IIIT colleges
    elif "IIIT" in name:
        return "IIIT"

    # Everything else
    else:
        return "GFTI"


# =====================================================
# STEP 1:
# Read final dataset
# =====================================================

df = pd.read_csv("josaa_final_dataset.csv")


# =====================================================
# STEP 2:
# Create new column
# =====================================================

df["College_Type"] = df["Institute"].apply(get_college_type)


# =====================================================
# STEP 3:
# Save updated dataset
# =====================================================

df.to_csv(
    "josaa_final_dataset.csv",
    index=False
)


# =====================================================
# STEP 4:
# Verify output
# =====================================================

print(df[["Institute", "College_Type"]].head(20))