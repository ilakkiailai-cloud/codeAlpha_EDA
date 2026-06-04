import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── STEP 1: Load Dataset ──────────────────────────────────
df = pd.read_csv("all_books.csv")

print("="*50)
print("1. DATA STRUCTURE")
print("="*50)
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print(f"\nColumn Names: {list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ── STEP 2: Clean Data ────────────────────────────────────
print("\n" + "="*50)
print("2. DATA CLEANING")
print("="*50)
df['price_clean'] = df['price'].str.replace('Â','').str.replace('£','').astype(float)
rating_map = {'One':1,'Two':2,'Three':3,'Four':4,'Five':5}
df['rating_num'] = df['rating'].map(rating_map)
print("Price and Rating columns cleaned!")
print(f"Missing values:\n{df.isnull().sum()}")

# ── STEP 3: Basic Statistics ──────────────────────────────
print("\n" + "="*50)
print("3. STATISTICS")
print("="*50)
print(f"Average Price: £{df['price_clean'].mean():.2f}")
print(f"Cheapest Book: £{df['price_clean'].min():.2f}")
print(f"Most Expensive: £{df['price_clean'].max():.2f}")
print(f"\nRating Distribution:\n{df['rating'].value_counts()}")

# ── STEP 4: Trends & Patterns ─────────────────────────────
print("\n" + "="*50)
print("4. TRENDS & PATTERNS")
print("="*50)
avg_price_by_rating = df.groupby('rating')['price_clean'].mean()
print(f"Average Price by Rating:\n{avg_price_by_rating}")

expensive = df[df['price_clean'] > 50]
print(f"\nBooks above £50: {len(expensive)}")
cheap = df[df['price_clean'] < 15]
print(f"Books below £15: {len(cheap)}")

# ── STEP 5: Visualizations ────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Books Dataset - EDA', fontsize=16)

# Chart 1: Rating Distribution
df['rating'].value_counts().plot(kind='bar', ax=axes[0,0], color='steelblue')
axes[0,0].set_title('Rating Distribution')
axes[0,0].set_xlabel('Rating')
axes[0,0].set_ylabel('Number of Books')

# Chart 2: Price Distribution
axes[0,1].hist(df['price_clean'], bins=20, color='orange', edgecolor='black')
axes[0,1].set_title('Price Distribution')
axes[0,1].set_xlabel('Price (£)')
axes[0,1].set_ylabel('Number of Books')

# Chart 3: Average Price by Rating
avg_price_by_rating.plot(kind='bar', ax=axes[1,0], color='green')
axes[1,0].set_title('Average Price by Rating')
axes[1,0].set_xlabel('Rating')
axes[1,0].set_ylabel('Average Price (£)')

# Chart 4: Books per Page
df['page'].value_counts().sort_index().plot(kind='line', ax=axes[1,1], color='red')
axes[1,1].set_title('Books per Page')
axes[1,1].set_xlabel('Page Number')
axes[1,1].set_ylabel('Number of Books')

plt.tight_layout()
plt.savefig('eda_charts.png')
print("\n✅ Charts saved as eda_charts.png!")

# ── STEP 6: Anomalies ─────────────────────────────────────
print("\n" + "="*50)
print("5. ANOMALIES DETECTED")
print("="*50)
q1 = df['price_clean'].quantile(0.25)
q3 = df['price_clean'].quantile(0.75)
iqr = q3 - q1
outliers = df[(df['price_clean'] < q1 - 1.5*iqr) | 
              (df['price_clean'] > q3 + 1.5*iqr)]
print(f"Price Outliers found: {len(outliers)}")
print(outliers[['title','price_clean','rating']].head())

print("\n✅ EDA COMPLETE!")
