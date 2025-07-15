import pandas as pd
import numpy as np
from scipy import stats


def generate_data(mean_diff, p_value, genotypes, output_file):
    n = sum(count for _, count in genotypes)
    if n < 2:
        raise ValueError("Total sample size must be at least 2")

    t_target = stats.t.ppf(1 - p_value / 2.0, df=n - 1)
    if t_target == 0:
        raise ValueError("Invalid p-value leading to t=0")

    sd_target = abs(mean_diff) * np.sqrt(n) / abs(t_target)
    baseline = np.random.normal(loc=100.0, scale=10.0, size=n)

    # generate differences with desired mean and sd
    diff = np.random.normal(loc=mean_diff, scale=sd_target, size=n)
    # adjust mean and sd exactly
    diff = diff - diff.mean() + mean_diff
    current_sd = diff.std(ddof=1)
    if current_sd == 0:
        diff = np.full(n, mean_diff)
    else:
        diff = (diff - diff.mean()) / current_sd * sd_target + mean_diff

    followup = baseline + diff
    genotype_list = []
    for name, count in genotypes:
        genotype_list.extend([name] * count)

    df = pd.DataFrame({
        "Genotype": genotype_list,
        "Baseline": baseline,
        "Followup": followup
    })

    df.to_csv(output_file, index=False)
    return df


def analyze_csv(file_path):
    """Calculate mean difference and paired t-test p-value from a CSV."""
    df = pd.read_csv(file_path)
    if not {"Baseline", "Followup"}.issubset(df.columns):
        raise ValueError("CSV must contain 'Baseline' and 'Followup' columns")

    diff = df["Followup"] - df["Baseline"]
    _, p_val = stats.ttest_rel(df["Followup"], df["Baseline"])
    return diff.mean(), p_val


def main():
    print("1) Generate new synthetic dataset")
    print("2) Analyze existing CSV")
    choice = input("Select option (1/2): ").strip() or "1"

    if choice == "2":
        file_path = input("Path to CSV file: ")
        mean_diff, p_val = analyze_csv(file_path)
        print(f"Mean difference: {mean_diff:.4f}")
        print(f"Paired t-test p-value: {p_val:.4g}")
        return

    mean_diff = float(input("Mean difference between followup and baseline: "))
    p_value = float(input("Target two-sided p-value: "))
    num_g = int(input("Number of genotypes: "))
    genotypes = []
    for i in range(num_g):
        name = input(f"Name for genotype {i+1}: ")
        count = int(input(f"Sample size for {name}: "))
        genotypes.append((name, count))
    output_file = input("Output CSV filename: ") or "output.csv"

    df = generate_data(mean_diff, p_value, genotypes, output_file)
    diff = df['Followup'] - df['Baseline']
    _, actual_p = stats.ttest_rel(df['Followup'], df['Baseline'])
    print(f"Generated {len(df)} samples. Mean difference: {diff.mean():.4f}")
    print(f"Paired t-test p-value: {actual_p:.4g}")
    print(f"Data saved to {output_file}")


if __name__ == "__main__":
    main()
