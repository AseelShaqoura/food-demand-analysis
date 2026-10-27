# %% [markdown]
# # تحليل طلب الوجبات الأسبوعي
#
# دفتر قابل لإعادة التشغيل من مجلد المشروع. يقرأ الملفات المحلية فقط، ويحافظ على المصادر كما هي، ويولد الرسوم وتقرير العميل.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr

ROOT = Path.cwd()
if not (ROOT / "data" / "train.csv").exists():
    try:
        ROOT = Path(__file__).resolve().parents[1]
    except NameError:
        ROOT = ROOT.parent
DATA = ROOT / "data"
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA / "train.csv"
CENTER_PATH = DATA / "fulfilment_center_info.csv"
MEAL_PATH = DATA / "meal_info.csv"

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 180,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "font.family": "DejaVu Sans",
})
COLORS = ["#146C94", "#19A7CE", "#F4A261", "#2A9D8F", "#E76F51", "#6C63FF"]

# %% [markdown]
## 1. تحميل البيانات والتحقق من سلامة المصدر

# %%
train = pd.read_csv(TRAIN_PATH)
centers = pd.read_csv(CENTER_PATH)
meals = pd.read_csv(MEAL_PATH)

source_rows = len(train)
source_orders = int(train["num_orders"].sum())
print(f"Training baseline: {source_rows:,} rows; {source_orders:,} orders")
print(f"Dimension sizes: {len(centers):,} centers; {len(meals):,} meals")
print(f"Training weeks: {train['week'].min()}–{train['week'].max()}")

# %% [markdown]
### فحص الجودة والربط

# %%
missing = pd.concat(
    {
        "train": train.isna().sum(),
        "centers": centers.isna().sum(),
        "meals": meals.isna().sum(),
    },
    axis=1,
).fillna(0).astype(int)
duplicate_rows = int(train.duplicated().sum())
duplicate_observation_keys = int(
    train.duplicated(["center_id", "meal_id", "week"]).sum()
)
duplicate_center_ids = int(centers["center_id"].duplicated().sum())
duplicate_meal_ids = int(meals["meal_id"].duplicated().sum())
unmatched_centers = int((~train["center_id"].isin(centers["center_id"])).sum())
unmatched_meals = int((~train["meal_id"].isin(meals["meal_id"])).sum())
price_above_base = int((train["checkout_price"] > train["base_price"]).sum())
q1, q3 = train["num_orders"].quantile([0.25, 0.75])
upper_fence = float(q3 + 1.5 * (q3 - q1))
high_order_count = int((train["num_orders"] > upper_fence).sum())
meal_week_coverage = train.groupby("meal_id")["week"].nunique()

quality = pd.DataFrame(
    [
        ("Missing cells", int(train.isna().sum().sum() + centers.isna().sum().sum() + meals.isna().sum().sum())),
        ("Exact duplicate training rows", duplicate_rows),
        ("Duplicate center-meal-week keys", duplicate_observation_keys),
        ("Duplicate center dimension IDs", duplicate_center_ids),
        ("Duplicate meal dimension IDs", duplicate_meal_ids),
        ("Training rows without center match", unmatched_centers),
        ("Training rows without meal match", unmatched_meals),
        ("Checkout price above base price", price_above_base),
        ("Orders above global Tukey upper fence", high_order_count),
    ],
    columns=["Check", "Count"],
)
print(quality.to_string(index=False))
print(
    "Weeks observed per meal:",
    int(meal_week_coverage.min()),
    "to",
    int(meal_week_coverage.max()),
    "of",
    int(train["week"].nunique()),
)
print("Test file is not used as labeled data: it has no num_orders target.")
assert duplicate_center_ids == duplicate_meal_ids == 0
assert unmatched_centers == unmatched_meals == 0

# %% [markdown]
## 2. ربط الأبعاد مع اختبارات صريحة

# %%
analysis = train.merge(
    centers, on="center_id", how="left", validate="many_to_one"
)
assert len(analysis) == source_rows
assert int(analysis["num_orders"].sum()) == source_orders
print(f"After center join: {len(analysis):,} rows; {int(analysis['num_orders'].sum()):,} orders")

analysis = analysis.merge(
    meals, on="meal_id", how="left", validate="many_to_one"
)
assert len(analysis) == source_rows
assert int(analysis["num_orders"].sum()) == source_orders
assert analysis[["center_type", "category", "cuisine"]].notna().all().all()
print(f"After meal join: {len(analysis):,} rows; {int(analysis['num_orders'].sum()):,} orders")

analysis["discount_rate"] = (
    analysis["base_price"] - analysis["checkout_price"]
) / analysis["base_price"]
analysis["revenue"] = analysis["checkout_price"] * analysis["num_orders"]

# %% [markdown]
## 3. مؤشرات الأداء والأنماط

# %%
weekly = analysis.groupby("week", as_index=False)["num_orders"].sum()
weekly.columns = ["week", "orders"]
category = (
    analysis.groupby("category", as_index=False)["num_orders"]
    .sum()
    .sort_values("num_orders", ascending=False)
)
category["share"] = category["num_orders"] / category["num_orders"].sum()
category["cumulative_share"] = category["share"].cumsum()
meal_summary = (
    analysis.groupby(["meal_id", "category", "cuisine"], as_index=False)
    .agg(total_orders=("num_orders", "sum"), row_mean=("num_orders", "mean"))
)
weekly_meal = analysis.groupby(["meal_id", "week"])["num_orders"].sum()
meal_cv = weekly_meal.groupby(level="meal_id").agg(["mean", "std", "count"])
meal_cv["cv"] = meal_cv["std"] / meal_cv["mean"]
meal_cv = meal_cv.rename(columns={"count": "observed_weeks"}).reset_index()
meal_summary = meal_summary.merge(meal_cv, on="meal_id", validate="one_to_one")
meal_summary["demand_volatility_priority"] = (
    meal_summary["cv"] * np.log1p(meal_summary["total_orders"])
)
meal_summary = meal_summary.sort_values(
    "demand_volatility_priority", ascending=False
)
center_summary = (
    analysis.groupby(["center_id", "center_type"], as_index=False)
    .agg(total_orders=("num_orders", "sum"), mean_orders_per_row=("num_orders", "mean"))
)
center_type = (
    analysis.groupby("center_type", as_index=False)
    .agg(
        centers=("center_id", "nunique"),
        total_orders=("num_orders", "sum"),
        mean_orders_per_row=("num_orders", "mean"),
    )
    .merge(
        center_summary.groupby("center_type", as_index=False)
        .agg(mean_center_orders=("total_orders", "mean")),
        on="center_type",
        validate="one_to_one",
    )
)
total_orders = int(analysis["num_orders"].sum())
average_weekly_orders = float(weekly["orders"].mean())
revenue_proxy = float(analysis["revenue"].sum())
mean_discount_rate = float(analysis["discount_rate"].mean())
email_rate = float(analysis["emailer_for_promotion"].mean())
homepage_rate = float(analysis["homepage_featured"].mean())

category_80_count = int((category["cumulative_share"] < 0.8).sum() + 1)
top_three_share = float(category.head(3)["num_orders"].sum() / total_orders)
print(f"Total orders: {total_orders:,}")
print(f"Average weekly orders: {average_weekly_orders:,.0f}")
print(f"Price-times-orders revenue proxy: {revenue_proxy:,.2f}")
print(f"Mean signed discount rate: {mean_discount_rate:.2%}")
print(f"Email promotion row rate: {email_rate:.2%}; homepage feature row rate: {homepage_rate:.2%}")
print(f"Top three category share: {top_three_share:.2%}")
print(f"Categories needed to exceed 80% of orders: {category_80_count}")
print(f"Meal weekly demand CV coverage: {meal_cv['observed_weeks'].min()}–{meal_cv['observed_weeks'].max()} observed weeks")
print("Center type summary:")
print(center_type.sort_values("mean_orders_per_row", ascending=False).to_string(index=False))

# %% [markdown]
## 4. مقارنة العروض والخصم إحصائيًا

# %%
def promotion_comparison(frame: pd.DataFrame, flag: str) -> dict:
    state_counts = frame.groupby(["center_id", "meal_id"])[flag].nunique()
    eligible = state_counts[state_counts == 2].index
    subset = frame.set_index(["center_id", "meal_id"]).loc[eligible].reset_index()
    promoted = subset.loc[subset[flag] == 1, "num_orders"]
    unpromoted = subset.loc[subset[flag] == 0, "num_orders"]
    test = mannwhitneyu(promoted, unpromoted, alternative="two-sided")
    median_difference = float(promoted.median() - unpromoted.median())
    return {
        "flag": flag,
        "eligible_strata": len(eligible),
        "n_promoted": len(promoted),
        "n_unpromoted": len(unpromoted),
        "median_promoted": float(promoted.median()),
        "median_unpromoted": float(unpromoted.median()),
        "median_difference": median_difference,
        "median_difference_pct": float(
            median_difference / unpromoted.median() * 100
        ),
        "mean_promoted": float(promoted.mean()),
        "mean_unpromoted": float(unpromoted.mean()),
        "u_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
    }


def p_value_label(value: float) -> str:
    return f"p<{0.001:.3f}" if value < 0.001 else f"p={value:.3g}"


promotion_results = [
    promotion_comparison(analysis, "emailer_for_promotion"),
    promotion_comparison(analysis, "homepage_featured"),
]
email_result = promotion_results[0]
home_result = promotion_results[1]
for result in promotion_results:
    print(
        f"{result['flag']}: {result['eligible_strata']:,} eligible center-meal strata; "
        f"n={result['n_promoted']:,}/{result['n_unpromoted']:,}; "
        f"median={result['median_promoted']:,.0f}/{result['median_unpromoted']:,.0f}; "
        f"median difference={result['median_difference_pct']:.1f}%; "
        f"Mann–Whitney U={result['u_statistic']:.3g}, "
        f"{p_value_label(result['p_value'])}"
    )

spearman = spearmanr(analysis["discount_rate"], analysis["num_orders"])
print(
    f"Spearman discount-orders: rho={spearman.statistic:.3f}, "
    f"{p_value_label(float(spearman.pvalue))}, n={len(analysis):,}"
)
print(
    "Interpretation: these are observational associations. Promotions may coincide "
    "with other factors, and repeated observations are not adjusted for dependence; "
    "neither test proves a causal effect."
)

# %% [markdown]
## 5. الرسوم الناتجة

# %%
def save_fig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close()


fig, ax = plt.subplots(figsize=(10, 4.8))
ax.plot(weekly["week"], weekly["orders"] / 1e6, color=COLORS[0], linewidth=2.3)
ax.fill_between(weekly["week"], weekly["orders"] / 1e6, color=COLORS[1], alpha=0.16)
ax.set(title="Observed orders fluctuate across 145 weeks", xlabel="Week index (no calendar dates)", ylabel="Orders (millions)")
ax.grid(axis="y", alpha=0.2)
save_fig("weekly_demand.png")

fig, ax = plt.subplots(figsize=(10, 5.2))
sorted_category = category.sort_values("num_orders", ascending=True)
ax.barh(sorted_category["category"], sorted_category["num_orders"] / 1e6, color=COLORS[0])
ax.axvline(category["num_orders"].sum() * 0.8 / 1e6, color=COLORS[2], linestyle="--", label="80% of all orders")
ax.set(title=f"{category_80_count} categories account for at least 80% of demand", xlabel="Orders (millions)", ylabel="")
ax.legend(frameon=False)
ax.grid(axis="x", alpha=0.2)
save_fig("category_concentration.png")

fig, ax = plt.subplots(figsize=(8, 4.8))
center_types_order = center_type.sort_values("mean_orders_per_row", ascending=False)
bars = ax.bar(center_types_order["center_type"], center_types_order["mean_orders_per_row"], color=COLORS[:len(center_types_order)])
ax.bar_label(bars, fmt="%.0f", padding=3)
ax.set(title="Average orders per row differ across center types", xlabel="Fulfilment center type", ylabel="Mean orders per observation")
ax.grid(axis="y", alpha=0.2)
save_fig("center_type_performance.png")

plot_rows = []
for result in promotion_results:
    flag = result["flag"]
    eligible = analysis.groupby(["center_id", "meal_id"])[flag].nunique()
    eligible = eligible[eligible == 2].index
    matched = analysis.set_index(["center_id", "meal_id"]).loc[eligible].reset_index()
    plot_rows.append((flag, matched.loc[matched[flag] == 0, "num_orders"].to_numpy(), matched.loc[matched[flag] == 1, "num_orders"].to_numpy()))
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
for ax, (flag, no, yes) in zip(axes, plot_rows):
    ax.boxplot([no, yes], tick_labels=["No", "Yes"], showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#B8DDE8"}, medianprops={"color": "#E76F51", "linewidth": 2})
    ax.set(title=f"Orders by {('email promotion' if flag.startswith('emailer') else 'homepage feature')}", xlabel="Promotion state", ylabel="Orders per observation")
    ax.grid(axis="y", alpha=0.2)
fig.suptitle("Promoted observations have higher medians in eligible center-meal strata", fontweight="bold")
save_fig("promotion_comparison.png")

sample = analysis.sample(n=min(12000, len(analysis)), random_state=17)
fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(sample["discount_rate"] * 100, sample["num_orders"], c=sample["homepage_featured"], cmap="coolwarm", alpha=0.35, s=12)
ax.set(title=f"Discount and demand move together weakly (Spearman ρ={spearman.statistic:.2f})", xlabel="Signed discount rate (%)", ylabel="Orders")
ax.grid(alpha=0.18)
ax.set_ylim(bottom=0)
fig.colorbar(scatter, ax=ax, label="Homepage featured (0/1)")
save_fig("discount_orders.png")

fig, ax = plt.subplots(figsize=(9, 5))
ranked = meal_summary.head(15).sort_values("demand_volatility_priority")
ax.barh(ranked["meal_id"].astype(str), ranked["demand_volatility_priority"], color=COLORS[4])
ax.set(title="A few meals combine high demand scale and high variability", xlabel="Priority score: weekly CV × log(1 + total orders)", ylabel="Meal ID")
ax.grid(axis="x", alpha=0.2)
save_fig("meal_volatility_priority.png")

print("Generated charts:", ", ".join(sorted(path.name for path in FIGURES.glob("*.png"))))

# %% [markdown]
## 6. توصيات قابلة للقياس وحدود الاستنتاج

# %%
top_meal = meal_summary.sort_values("total_orders", ascending=False).iloc[0]
highest_cv = meal_summary.sort_values("cv", ascending=False).iloc[0]
highest_priority = meal_summary.iloc[0]
top_center_type = center_type.sort_values("mean_orders_per_row", ascending=False).iloc[0]
email_result = promotion_results[0]
home_result = promotion_results[1]
recommendations = [
    {
        "action": "ابدأ تجربة توقع أسبوعية منفصلة لأعلى فئات الطلب، دون تغيير سياسة الشراء مباشرة.",
        "evidence": f"أعلى {category_80_count} فئات تتجاوز 80% من الطلب؛ Beverage هي الأكبر ({category.iloc[0]['share']:.1%}).",
        "metric": "MAE أسبوعي مقابل التوقع الموسمي الساذج على فترة زمنية محجوزة.",
    },
    {
        "action": f"اختبر احتياطيًا مرنًا للوجبة {int(highest_priority['meal_id'])} وفق مؤشر الحجم والتذبذب، مع متابعة أسبوعية.",
        "evidence": f"أعلى أولوية مركبة: CV={highest_priority['cv']:.2f} وحجم {int(highest_priority['total_orders']):,}؛ أعلى CV منفرد هو {highest_cv['cv']:.2f} للوجبة {int(highest_cv['meal_id'])}. المؤشر وصفي ولا يقيس النفاد.",
        "metric": "نسبة النفاد والهدر أسبوعيًا، بعد جمع بيانات مخزون/نفاد مستقبلية.",
    },
    {
        "action": "نفّذ تجربة عشوائية أو طرحًا مرحليًا للعروض قبل زيادة الإنفاق عليها.",
        "evidence": f"وسيط الطلب أعلى رصديًا بـ {email_result['median_difference_pct']:.1f}% للبريد و{home_result['median_difference_pct']:.1f}% للظهور، لكن الاختبار غير سببي.",
        "metric": "الزيادة الصافية في الطلب/الهامش بين المجموعات المعالجة والضابطة.",
    },
]
for i, item in enumerate(recommendations, start=1):
    print(f"{i}. {item['action']}\n   Evidence: {item['evidence']}\n   Success metric: {item['metric']}")
print(
    "Limits: no calendar dates; coded cities; absent meal-week rows are not known zeros; "
    "promotions are observational and not randomized; no stockout/waste/inventory field; "
    "revenue proxy excludes costs and margin."
)

# %% [markdown]
## 7. إنشاء تقرير العميل PDF

# %%
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
import arabic_reshaper
from bidi.algorithm import get_display

REPORT_PATH = ROOT / "reports" / "client_report.pdf"
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\tahoma.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]
font_path = next((p for p in FONT_CANDIDATES if p.exists()), None)
if font_path is None:
    raise FileNotFoundError("No supported TTF font found for the Arabic PDF.")
pdfmetrics.registerFont(TTFont("ProjectArabic", str(font_path)))

PAGE_W, PAGE_H = A4
NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#146C94")
TEAL = colors.HexColor("#19A7CE")
GOLD = colors.HexColor("#F4A261")
INK = colors.HexColor("#243B53")
MUTED = colors.HexColor("#627D98")
PALE = colors.HexColor("#F0F6FA")
WHITE = colors.white


def rtl(text: str) -> str:
    return get_display(arabic_reshaper.reshape(str(text)))


pdf = canvas.Canvas(str(REPORT_PATH), pagesize=A4)
pdf.setTitle("تحليل الطلب الأسبوعي - تقرير العميل")
pdf.setAuthor("Food Demand Analysis")


def page_header(title: str, section: str) -> None:
    pdf.setFillColor(NAVY)
    pdf.rect(0, PAGE_H - 70, PAGE_W, 70, fill=1, stroke=0)
    pdf.setFillColor(TEAL)
    pdf.rect(0, PAGE_H - 74, PAGE_W, 4, fill=1, stroke=0)
    pdf.setFillColor(WHITE)
    pdf.setFont("ProjectArabic", 18)
    pdf.drawRightString(PAGE_W - 42, PAGE_H - 42, rtl(title))
    pdf.setFillColor(MUTED)
    pdf.setFont("ProjectArabic", 8)
    pdf.drawString(42, 27, rtl("تحليل وصفي لبيانات مجهولة الهوية | لا توجد تواريخ تقويمية"))
    pdf.drawRightString(PAGE_W - 42, 27, section)
    pdf.setFillColor(MUTED)
    pdf.setFont("Helvetica", 8)
    pdf.drawCentredString(PAGE_W / 2, 27, str(pdf.getPageNumber()))


def paragraph(text: str, x: float, y: float, max_width: float, size: int = 11, leading: int = 18, color=INK) -> float:
    pdf.setFont("ProjectArabic", size)
    pdf.setFillColor(color)
    words = str(text).split()
    lines, line = [], ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if pdfmetrics.stringWidth(rtl(candidate), "ProjectArabic", size) > max_width and line:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    for line in lines:
        pdf.drawRightString(x + max_width, y, rtl(line))
        y -= leading
    return y


def section_label(text: str, x: float, y: float, max_width: float) -> None:
    pdf.setFillColor(PALE)
    pdf.roundRect(x, y - 10, max_width, 26, 7, fill=1, stroke=0)
    pdf.setFillColor(BLUE)
    pdf.setFont("ProjectArabic", 11)
    pdf.drawRightString(x + max_width - 12, y - 2, rtl(text))


def bullet(text: str, x: float, y: float, max_width: float, size: int = 10) -> float:
    pdf.setFillColor(GOLD)
    pdf.circle(x + max_width - 4, y + 3, 3, fill=1, stroke=0)
    return paragraph(text, x + 3, y, max_width - 16, size=size, leading=16)


def add_chart(filename: str, x: float, y: float, width: float, height: float) -> None:
    path = FIGURES / filename
    if not path.exists():
        raise FileNotFoundError(f"Expected chart was not generated: {path}")
    pdf.drawImage(str(path), x, y, width=width, height=height, preserveAspectRatio=True, anchor="c", mask="auto")


# Cover and executive summary
pdf.setFillColor(NAVY)
pdf.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
pdf.setFillColor(TEAL)
pdf.rect(0, PAGE_H - 15, PAGE_W, 15, fill=1, stroke=0)
pdf.setFillColor(GOLD)
pdf.circle(PAGE_W - 70, PAGE_H - 85, 32, fill=1, stroke=0)
pdf.setFillColor(WHITE)
pdf.setFont("ProjectArabic", 26)
pdf.drawRightString(PAGE_W - 46, PAGE_H - 180, rtl("تحليل الطلب الأسبوعي"))
pdf.setFont("ProjectArabic", 15)
pdf.drawRightString(PAGE_W - 46, PAGE_H - 215, rtl("رؤى عملية لتخطيط شراء الوجبات"))
pdf.setFillColor(colors.HexColor("#B8DDE8"))
pdf.setFont("ProjectArabic", 11)
pdf.drawRightString(PAGE_W - 46, PAGE_H - 250, rtl("تقرير تنفيذي موجز + ملخص عرض"))
pdf.setFillColor(WHITE)
pdf.roundRect(44, PAGE_H - 440, PAGE_W - 88, 125, 14, fill=1, stroke=0)
pdf.setFillColor(NAVY)
pdf.setFont("ProjectArabic", 13)
pdf.drawRightString(PAGE_W - 62, PAGE_H - 340, rtl("الخلاصة في خمسة أسطر"))
summary_lines = [
    f"أكبر ثلاث فئات تمثل {top_three_share:.1%} من الطلب؛ وخمس فئات تتجاوز 80%.",
    f"إجمالي التدريب {total_orders:,} طلب، بمتوسط {average_weekly_orders:,.0f} طلب أسبوعيًا.",
    f"ارتبط البريد بفرق وسيط {email_result['median_difference_pct']:.1f}% والظهور بـ {home_result['median_difference_pct']:.1f}%.",
    f"متوسط الخصم الموقّع {mean_discount_rate:.2%}؛ الإيراد السعري التقديري {revenue_proxy / 1e9:.2f} مليار وحدة سعر.",
    "هذه نتائج رصدية؛ لا تثبت أن العرض أو الخصم سبب زيادة الطلب.",
]
for row, line in enumerate(summary_lines):
    pdf.setFillColor(TEAL if row == 0 else INK)
    pdf.setFont("ProjectArabic", 8)
    pdf.drawRightString(PAGE_W - 60, PAGE_H - 363 - row * 13, rtl(line))
pdf.setFillColor(colors.HexColor("#B8DDE8"))
pdf.setFont("Helvetica", 10)
pdf.drawString(46, 70, f"{source_rows:,} rows  |  {total_orders:,} orders  |  145 weeks")
pdf.setFont("ProjectArabic", 9)
pdf.drawRightString(PAGE_W - 46, 70, rtl("البيانات محلية ومجهولة الهوية؛ لم تُرفع إلى الإنترنت"))
pdf.showPage()

# Executive findings and chart
page_header("أين يتركز الطلب؟", "01 | الطلب")
section_label("النتائج الرئيسية", 42, PAGE_H - 102, PAGE_W - 84)
y = PAGE_H - 142
y = bullet(f"سجلت البيانات {total_orders:,} طلبًا خلال 145 أسبوعًا؛ المتوسط {average_weekly_orders:,.0f} طلب أسبوعيًا.", 48, y, PAGE_W - 96)
y = bullet(f"تتجاوز أول {category_80_count} فئات معًا 80% من الطلب. أكبر ثلاث فئات تمثل {top_three_share:.1%}.", 48, y - 8, PAGE_W - 96)
y = bullet(f"الفئة الأكبر هي {category.iloc[0]['category']} بحصة {category.iloc[0]['share']:.1%} من الطلب.", 48, y - 8, PAGE_W - 96)
y = bullet(f"الإيراد السعري التقديري {revenue_proxy / 1e9:.2f} مليار وحدة سعر؛ ومتوسط الخصم الموقّع {mean_discount_rate:.2%}.", 48, y - 8, PAGE_W - 96)
section_label("تركيز الطلب حسب الفئة", 42, y - 12, PAGE_W - 84)
add_chart("category_concentration.png", 45, 70, PAGE_W - 90, 280)
pdf.showPage()

# Weekly demand and quality
page_header("تذبذب أسبوعي وجودة بيانات موثقة", "02 | الجودة")
section_label("تغير الطلب", 42, PAGE_H - 102, PAGE_W - 84)
add_chart("weekly_demand.png", 48, PAGE_H - 350, PAGE_W - 96, 220)
paragraph(
    f"يتراوح الإجمالي الأسبوعي من {int(weekly['orders'].min()):,} إلى {int(weekly['orders'].max()):,} طلب. لا تحمل أرقام الأسابيع تواريخ تقويمية، لذا لا نفسرها كمواسم أو أحداث محددة.",
    55, PAGE_H - 380, PAGE_W - 110, size=10, leading=17
)
section_label("فحوص المصدر وقرار المعالجة", 42, PAGE_H - 430, PAGE_W - 84)
quality_lines = [
    "لا قيم مفقودة، ولا صفوف تدريب مكررة، وجميع مفاتيح المركز والوجبة لها مطابقات.",
    "ثبتت عمليات الربط كـ many-to-one؛ لم يتغير عدد الصفوف أو مجموع الطلب بعد الربط.",
    f"سعر الشراء أعلى من السعر الأساسي في {price_above_base:,} صف؛ احتُفظ بها ولم تُصنف تلقائيًا كخطأ.",
    f"{high_order_count:,} صف يتجاوز حد Tukey؛ احتُفظ بها كقمم محتملة. تغطية الوجبات بين {meal_week_coverage.min()} و{meal_week_coverage.max()} أسبوعًا.",
]
y = PAGE_H - 470
for line in quality_lines:
    y = bullet(line, 48, y, PAGE_W - 96, size=9) - 6
pdf.showPage()

# Centers and products
page_header("اختلاف أداء المراكز والوجبات", "03 | التشغيل")
section_label("متوسط الطلب حسب نوع المركز", 42, PAGE_H - 102, PAGE_W - 84)
add_chart("center_type_performance.png", 52, PAGE_H - 358, PAGE_W - 104, 220)
paragraph(
    f"أعلى متوسط طلب لكل صف هو {top_center_type['center_type']} ({top_center_type['mean_orders_per_row']:.0f}). هذا وصف لصفوف الطلب المرصودة، وليس مقياسًا لكفاءة المركز؛ أنواع المراكز وأحجامها مختلفة.",
    55, PAGE_H - 390, PAGE_W - 110, size=10, leading=17
)
section_label("مزيج حجم الطلب والتذبذب", 42, PAGE_H - 440, PAGE_W - 84)
add_chart("meal_volatility_priority.png", 52, 70, PAGE_W - 104, 250)
paragraph(
    f"الوجبة {int(top_meal['meal_id'])} صاحبة أعلى إجمالي مرصود ({int(top_meal['total_orders']):,})، بينما أعلى معامل تذبذب أسبوعي مرصود هو {meal_summary['cv'].max():.2f}. مؤشر الأولوية يجمع الحجم مع CV؛ لا يقيس نفاد المخزون.",
    55, PAGE_H - 475, PAGE_W - 110, size=9, leading=15
)
pdf.showPage()

# Promotions and prices
page_header("العروض والخصومات: ارتباط لا سببية", "04 | التسويق")
section_label("مقارنة الوسيط في الصفوف المؤهلة", 42, PAGE_H - 102, PAGE_W - 84)
add_chart("promotion_comparison.png", 48, PAGE_H - 365, PAGE_W - 96, 220)
y = PAGE_H - 392
y = bullet(
    f"البريد الترويجي: وسيط {email_result['median_promoted']:.0f} مقابل {email_result['median_unpromoted']:.0f} طلبًا ({email_result['median_difference_pct']:.1f}% فرقًا نسبيًا)، n={email_result['n_promoted']:,}/{email_result['n_unpromoted']:,}؛ Mann–Whitney p<0.001.",
    48, y, PAGE_W - 96, size=9
)
y = bullet(
    f"الظهور في الصفحة الرئيسية: وسيط {home_result['median_promoted']:.0f} مقابل {home_result['median_unpromoted']:.0f} ({home_result['median_difference_pct']:.1f}%)؛ Mann–Whitney p<0.001.",
    48, y - 7, PAGE_W - 96, size=9
)
section_label("علاقة الخصم بالطلب", 42, y - 14, PAGE_W - 84)
add_chart("discount_orders.png", 52, 62, PAGE_W - 104, 205)
paragraph(
    f"ارتباط Spearman بين معدل الخصم والطلب rho={spearman.statistic:.3f} (p<0.001، n={len(analysis):,})؛ الارتباط موجب لكنه ضعيف. اختبار الرتب لا يعزل أثر السعر عن بقية العوامل.",
    55, y - 42, PAGE_W - 110, size=9, leading=15
)
pdf.showPage()

# Recommendations and short presentation page
page_header("ثلاث خطوات قابلة للقياس", "05 | القرار")
for index, item in enumerate(recommendations, start=1):
    top = PAGE_H - 112 - (index - 1) * 145
    pdf.setFillColor(PALE)
    pdf.roundRect(45, top - 112, PAGE_W - 90, 122, 12, fill=1, stroke=0)
    pdf.setFillColor(BLUE)
    pdf.circle(PAGE_W - 67, top - 25, 14, fill=1, stroke=0)
    pdf.setFillColor(WHITE)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawCentredString(PAGE_W - 67, top - 29, str(index))
    y = paragraph(item["action"], 70, top - 27, PAGE_W - 150, size=10, leading=16)
    y = paragraph(f"الدليل: {item['evidence']}", 70, y - 5, PAGE_W - 150, size=8, leading=13, color=MUTED)
    paragraph(f"مقياس النجاح: {item['metric']}", 70, y - 4, PAGE_W - 150, size=8, leading=13, color=INK)
section_label("ملخص العرض في دقيقة", 42, PAGE_H - 560, PAGE_W - 84)
y = PAGE_H - 595
for line in [
    "المشكلة: الطلب المتقلب يجعل الشراء الأسبوعي عرضة للهدر أو نفاد الوجبات.",
    "الدليل: خمس فئات تغطي أكثر من 80%؛ فروق العروض ملحوظة لكنها غير سببية.",
    "القرار: اختبر توقعًا أسبوعيًا وتجارب ترويج مضبوطة، وقِس الهدر والنفاد قبل تعميم أي تغيير.",
    "استخدام الذكاء الاصطناعي: مساعدة في صياغة التحليل؛ النتائج تحققت بتشغيل محلي للبيانات.",
]:
    y = bullet(line, 48, y, PAGE_W - 96, size=9) - 7
pdf.showPage()

# Limitations and methods
page_header("ما الذي نعرفه وما الذي لا تثبته البيانات؟", "06 | الحدود")
section_label("المنهج", 42, PAGE_H - 102, PAGE_W - 84)
y = PAGE_H - 140
for line in [
    "مصادر محلية: سجل التدريب، وبيانات المراكز والوجبات؛ ربط مفاتيح فريد مع تحقق صريح من ثبات الصفوف ومجموع الطلب.",
    "مقارنة البريد والظهور: Mann–Whitney ثنائي الطرف، بعد حصر المقارنة في تركيبات مركز-وجبة التي ظهر فيها الحالتان.",
    "الاختبار يجمع الصفوف المؤهلة ولا يضبط الاعتماد بين الأسابيع المتكررة داخل تركيبة المركز-الوجبة.",
    "الخصم: (السعر الأساسي − سعر الشراء) ÷ السعر الأساسي؛ علاقته بالطلب عبر Spearman.",
    "الإيراد التقديري: سعر الشراء × عدد الطلبات؛ لا يمثل الربح ولا يتضمن التكلفة.",
]:
    y = bullet(line, 48, y, PAGE_W - 96, size=9) - 5
section_label("حدود مهمة", 42, y - 4, PAGE_W - 84)
y -= 45
for line in [
    "لا تواريخ فعلية أو تعريف للمواسم؛ رموز المدن لا تكشف مواقعها.",
    "لا بيانات مخزون أو نفاد أو هدر؛ لا يمكن إثبات أين حدثت هذه النتائج أو مقدارها.",
    "تخصيص العروض رصدي وغير عشوائي، كما قد تتزامن العروض مع السعر والظهور؛ لا يمكن استنتاج السببية.",
    "غياب سجل وجبة في أسبوع لا يعني بالضرورة أن الطلب صفر.",
    "قيم p الصغيرة لا تضمن أهمية تجارية، والاختبارات لا تعالج كل التحيزات أو تعدد المقارنات.",
]:
    y = bullet(line, 48, y, PAGE_W - 96, size=9) - 5
paragraph("الخطوة التالية الأفضل: اختبار عشوائي أو طرح مرحلي، مع جمع بيانات الهدر والنفاد والهامش وتواريخ فعلية.", 52, 74, PAGE_W - 104, size=10, leading=17, color=BLUE)
pdf.save()
print(f"Client PDF saved: {REPORT_PATH}")
