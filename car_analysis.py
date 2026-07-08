"""
🚗 Car Price Analysis — Python Data Analysis Project
======================================================
Dataset  : 500 Indian Used/New Cars
Libraries: Pandas, Matplotlib, Seaborn, NumPy
Charts   : 8 Publication-ready visualizations
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings, os

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────
BASE   = os.path.dirname(__file__)
DATA   = os.path.join(BASE, "car_prices.csv")
CHARTS = os.path.join(BASE, "charts")
os.makedirs(CHARTS, exist_ok=True)

# ── Style ───────────────────────────────────────────────────────
PALETTE = ["#2D6A4F","#40916C","#52B788","#74C69D","#95D5B2","#B7E4C7","#D8F3DC","#1B4332","#081C15","#F4A261"]
BG = "#F8F9FA"
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor":   BG,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
})

def save(name):
    path = os.path.join(CHARTS, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"  ✅ {name}")

# ══════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN DATA
# ══════════════════════════════════════════════════════════════
print("\n📦 Loading data...")
df = pd.read_csv(DATA)

print(f"  Shape     : {df.shape}")
print(f"  Nulls     : {df.isnull().sum().sum()}")
print(f"  Duplicates: {df.duplicated().sum()}")
print(f"\n  Price Range : ₹{df['price'].min():,.0f} — ₹{df['price'].max():,.0f}")
print(f"  Avg Price   : ₹{df['price'].mean():,.0f}")
print(f"  Brands      : {df['brand'].nunique()}")
print(f"  Models      : {df['model'].nunique()}")

df["price_lakh"] = (df["price"] / 100000).round(2)
df["car_age"]    = 2024 - df["year"]

# ══════════════════════════════════════════════════════════════
# 2. CHART 1 — OVERVIEW DASHBOARD
# ══════════════════════════════════════════════════════════════
print("\n📊 Generating charts...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("🚗 Car Price Analysis — Overview Dashboard", fontsize=16, fontweight="bold", y=1.01)

# 2a. Brand-wise Avg Price
brand_price = df.groupby("brand")["price_lakh"].mean().sort_values(ascending=True)
axes[0,0].barh(brand_price.index, brand_price.values, color=PALETTE[:len(brand_price)])
axes[0,0].set_title("Avg Price by Brand (₹ Lakh)")
axes[0,0].set_xlabel("Avg Price (₹ L)")
for bar in axes[0,0].patches:
    axes[0,0].text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                   f"₹{bar.get_width():.1f}L", va="center", fontsize=8)

# 2b. Fuel Type Distribution
fuel_counts = df["fuel_type"].value_counts()
axes[0,1].pie(fuel_counts.values, labels=fuel_counts.index,
              autopct="%1.1f%%", colors=PALETTE, startangle=140,
              wedgeprops={"edgecolor":"white","linewidth":2})
axes[0,1].set_title("Fuel Type Distribution")

# 2c. Price Distribution
axes[0,2].hist(df["price_lakh"], bins=25, color=PALETTE[1], edgecolor="white", linewidth=0.8)
axes[0,2].axvline(df["price_lakh"].mean(), color="red", linestyle="--",
                  label=f"Mean: ₹{df['price_lakh'].mean():.1f}L")
axes[0,2].axvline(df["price_lakh"].median(), color="orange", linestyle="--",
                  label=f"Median: ₹{df['price_lakh'].median():.1f}L")
axes[0,2].set_title("Price Distribution")
axes[0,2].set_xlabel("Price (₹ Lakh)")
axes[0,2].legend()

# 2d. Transmission Distribution
trans_counts = df["transmission"].value_counts()
axes[1,0].bar(trans_counts.index, trans_counts.values, color=PALETTE[2:7], edgecolor="white")
axes[1,0].set_title("Transmission Type Count")
axes[1,0].set_ylabel("Number of Cars")
for bar in axes[1,0].patches:
    axes[1,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                   str(int(bar.get_height())), ha="center", fontsize=9)

# 2e. Body Type vs Avg Price
body_price = df.groupby("body_type")["price_lakh"].mean().sort_values(ascending=False)
axes[1,1].bar(body_price.index, body_price.values, color=PALETTE[3:9], edgecolor="white")
axes[1,1].set_title("Avg Price by Body Type (₹ Lakh)")
axes[1,1].set_ylabel("Avg Price (₹ L)")
axes[1,1].tick_params(axis="x", rotation=20)

# 2f. Year-wise Cars Count
year_count = df["year"].value_counts().sort_index()
axes[1,2].bar(year_count.index, year_count.values, color=PALETTE[0], edgecolor="white")
axes[1,2].set_title("Cars by Manufacturing Year")
axes[1,2].set_xlabel("Year")
axes[1,2].set_ylabel("Count")

plt.tight_layout()
save("01_overview_dashboard.png")

# ══════════════════════════════════════════════════════════════
# 3. CHART 2 — PRICE ANALYSIS
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("💰 Car Price Analysis", fontsize=15, fontweight="bold")

# Brand boxplot
brand_order = df.groupby("brand")["price_lakh"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="brand", y="price_lakh", order=brand_order,
            palette=PALETTE, ax=axes[0])
axes[0].set_title("Price Distribution by Brand")
axes[0].set_xlabel("Brand")
axes[0].set_ylabel("Price (₹ Lakh)")
axes[0].tick_params(axis="x", rotation=45)

# Fuel type vs Price
sns.boxplot(data=df, x="fuel_type", y="price_lakh", palette=PALETTE, ax=axes[1])
axes[1].set_title("Price by Fuel Type")
axes[1].set_xlabel("Fuel Type")
axes[1].set_ylabel("Price (₹ Lakh)")
axes[1].tick_params(axis="x", rotation=20)

# Condition vs Price
cond_order = ["Excellent","Good","Fair","Poor"]
sns.boxplot(data=df, x="condition", y="price_lakh", order=cond_order,
            palette=["#2D6A4F","#52B788","#F4A261","#E63946"], ax=axes[2])
axes[2].set_title("Price by Car Condition")
axes[2].set_xlabel("Condition")
axes[2].set_ylabel("Price (₹ Lakh)")

plt.tight_layout()
save("02_price_analysis.png")

# ══════════════════════════════════════════════════════════════
# 4. CHART 3 — KMS DRIVEN & DEPRECIATION
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("📉 Depreciation & KMs Driven Analysis", fontsize=15, fontweight="bold")

# Scatter: kms vs price
sc = axes[0].scatter(df["kms_driven"]/1000, df["price_lakh"],
                     c=df["car_age"], cmap="RdYlGn_r", alpha=0.6, s=40)
plt.colorbar(sc, ax=axes[0], label="Car Age (Years)")
axes[0].set_title("KMs Driven vs Price (colored by Age)")
axes[0].set_xlabel("KMs Driven (000s)")
axes[0].set_ylabel("Price (₹ Lakh)")

# Car age vs avg price
age_price = df.groupby("car_age")["price_lakh"].mean().reset_index()
axes[1].plot(age_price["car_age"], age_price["price_lakh"],
             marker="o", color=PALETTE[0], linewidth=2.5, markersize=7)
axes[1].fill_between(age_price["car_age"], age_price["price_lakh"],
                     alpha=0.15, color=PALETTE[0])
axes[1].set_title("Car Age vs Avg Price (Depreciation Curve)")
axes[1].set_xlabel("Car Age (Years)")
axes[1].set_ylabel("Avg Price (₹ Lakh)")

plt.tight_layout()
save("03_depreciation_analysis.png")

# ══════════════════════════════════════════════════════════════
# 5. CHART 4 — BRAND & MODEL ANALYSIS
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("🏷️ Brand & Model Popularity", fontsize=15, fontweight="bold")

# Brand count
brand_count = df["brand"].value_counts()
axes[0].bar(brand_count.index, brand_count.values, color=PALETTE, edgecolor="white")
axes[0].set_title("Cars Count by Brand")
axes[0].set_ylabel("Count")
axes[0].tick_params(axis="x", rotation=45)
for bar in axes[0].patches:
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 str(int(bar.get_height())), ha="center", fontsize=8)

# Top 15 models
top_models = df["model"].value_counts().head(15).sort_values()
axes[1].barh(top_models.index, top_models.values, color=PALETTE[2])
axes[1].set_title("Top 15 Most Listed Models")
axes[1].set_xlabel("Count")

plt.tight_layout()
save("04_brand_model_analysis.png")

# ══════════════════════════════════════════════════════════════
# 6. CHART 5 — CITY & SELLER ANALYSIS
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("🌆 City & Seller Type Analysis", fontsize=15, fontweight="bold")

# City-wise avg price
city_price = df.groupby("city")["price_lakh"].mean().sort_values(ascending=False)
axes[0].bar(city_price.index, city_price.values, color=PALETTE[1], edgecolor="white")
axes[0].set_title("City-wise Avg Car Price (₹ Lakh)")
axes[0].set_ylabel("Avg Price (₹ L)")
axes[0].tick_params(axis="x", rotation=45)
for bar in axes[0].patches:
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                 f"₹{bar.get_height():.1f}L", ha="center", fontsize=8)

# Seller type
seller_data = df.groupby("seller_type")["price_lakh"].mean()
axes[1].bar(seller_data.index, seller_data.values,
            color=["#2D6A4F","#52B788","#95D5B2"], edgecolor="white", width=0.5)
axes[1].set_title("Avg Price by Seller Type")
axes[1].set_ylabel("Avg Price (₹ Lakh)")
for bar in axes[1].patches:
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                 f"₹{bar.get_height():.1f}L", ha="center", fontsize=10)

plt.tight_layout()
save("05_city_seller_analysis.png")

# ══════════════════════════════════════════════════════════════
# 7. CHART 6 — OWNERS & MILEAGE
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("👥 Owners & Mileage Analysis", fontsize=15, fontweight="bold")

# Owners vs price
owners_price = df.groupby("owners")["price_lakh"].mean()
axes[0].bar(["1st Owner","2nd Owner","3rd Owner","4th+ Owner"],
            owners_price.values, color=PALETTE[:4], edgecolor="white", width=0.5)
axes[0].set_title("Avg Price by Number of Owners")
axes[0].set_ylabel("Avg Price (₹ Lakh)")
for bar in axes[0].patches:
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                 f"₹{bar.get_height():.1f}L", ha="center", fontsize=10)

# Mileage distribution
axes[1].hist(df["mileage_kmpl"], bins=20, color=PALETTE[3], edgecolor="white")
axes[1].axvline(df["mileage_kmpl"].mean(), color="red", linestyle="--",
                label=f"Avg: {df['mileage_kmpl'].mean():.1f} kmpl")
axes[1].set_title("Mileage Distribution (kmpl)")
axes[1].set_xlabel("Mileage (kmpl)")
axes[1].set_ylabel("Count")
axes[1].legend()

plt.tight_layout()
save("06_owners_mileage_analysis.png")

# ══════════════════════════════════════════════════════════════
# 8. CHART 7 — CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 7))
num_cols = ["price_lakh","car_age","kms_driven","engine_cc","mileage_kmpl","owners","seats"]
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", ax=ax, linewidths=0.5,
            cbar_kws={"shrink":0.8})
ax.set_title("Numerical Features — Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
save("07_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════
# 9. KEY METRICS
# ══════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("📊 KEY METRICS SUMMARY")
print("="*55)
print(f"  Total Cars        : {len(df)}")
print(f"  Total Brands      : {df['brand'].nunique()}")
print(f"  Total Models      : {df['model'].nunique()}")
print(f"  Price Range       : ₹{df['price_lakh'].min():.1f}L — ₹{df['price_lakh'].max():.1f}L")
print(f"  Avg Price         : ₹{df['price_lakh'].mean():.2f} Lakh")
print(f"  Median Price      : ₹{df['price_lakh'].median():.2f} Lakh")
print(f"  Most Listed Brand : {df['brand'].value_counts().idxmax()}")
print(f"  Most Popular Fuel : {df['fuel_type'].value_counts().idxmax()}")
print(f"  Avg Mileage       : {df['mileage_kmpl'].mean():.1f} kmpl")
print(f"  Avg KMs Driven    : {df['kms_driven'].mean():,.0f} kms")
print(f"  1st Owner Cars    : {(df['owners']==1).sum()} ({(df['owners']==1).mean()*100:.1f}%)")
print("="*55)
print("\n✅ All 7 charts saved to /charts folder!")
