# CrashLens Final v6 — نسخه نهایی یکپارچه

این نسخه، داشبورد Computer Vision پروژه Nexar را با داشبورد اصلی تیم NLP (تحلیل نظرات محصولات آمازون) در یک رابط Streamlit ادغام می‌کند. تعداد صفحات اصلی به **۵ صفحه** رسیده است: ۴ صفحه برای Computer Vision و ۱ صفحه برای NLP. داشبورد NLP درون صفحه خودش ۸ زبانه تعاملی اصلی تیم را حفظ می‌کند.

## اجرای داشبورد در ویندوز

ZIP را Extract کنید. داخل پوشه `CrashLens_Final_v6` یک CMD باز کنید و اجرا کنید:

```bat
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt --default-timeout=120 --retries=10
.venv\Scripts\python.exe -m streamlit run app.py
```

اگر قبلاً محیط مجازی ساخته‌اید، مرحله اول لازم نیست. برای اجرای سریع‌تر بعد از نصب کتابخانه‌ها:

```bat
.venv\Scripts\python.exe -m streamlit run app.py
```

آدرس محلی داشبورد: `http://localhost:8501`

فایل `run_windows.bat` هم می‌تواند از محیط مجازی داخل پوشه پروژه یا `%USERPROFILE%\\.venv` استفاده کند.

## صفحات اصلی

| صفحه | چه چیزی نمایش می‌دهد؟ |
|---|---|
| **CV Overview & Architecture** | خلاصه پروژه Nexar، نتایج ثبت‌شده، درخت معماری Detection و Anticipation و نمودارهای EDA |
| **Video Demo · Two Models** | آپلود ویدیو و اجرای واقعی یکی یا هر دو مدل، در صورت وجود دو Checkpoint صحیح |
| **Experiments & Data Audit** | گراف مسیر آزمایش‌ها، جزئیات لاگ، روند آموزش، Ablation و بررسی Data Leakage |
| **CV Evaluation & Errors** | نتایج ارزیابی، Confusion Matrix و بررسی خطاها در صورت بارگذاری فایل امتیاز نمونه‌ها |
| **NLP Dashboard & Run Model** | داشبورد اصلی تیم NLP با ۸ زبانه، به‌علاوه محیط آموزش و اجرای مدل کلاسیک روی متن جدید |

### هشت زبانه داشبورد اصلی NLP

1. **Overview:** خلاصه داده، محصولات، امتیازها و نتیجه مدل.
2. **Ratings:** توزیع امتیازها، روند زمانی و مقایسه توزیع آموزش با تخمین آزمون.
3. **Words:** کلمات پرتکرار نظرات مثبت، خنثی و منفی.
4. **Reviewers & Votes:** نظردهندگان دارای بیشترین رأی و میزان تمرکز رأی‌ها.
5. **Text Length:** توزیع طول متن و بررسی سقف کاراکتر.
6. **Products & Brands:** محصولات و برندهای پرتکرار و میانگین امتیاز آن‌ها.
7. **Warranty:** رضایت ثبت‌شده در نظراتی که درباره ضمانت صحبت می‌کنند.
8. **Model:** مقایسه Classical، Transformer و Ensemble، همراه با معیارها و ماتریس درهم‌ریختگی.

این داشبورد از فایل HTML اصلی تیم استفاده می‌کند. داده‌های نمودارها داخل همان فایل قرار دارند؛ برای مشاهده این ۸ زبانه به فایل خام دیتاست یا اینترنت نیاز نیست.

## اجرای مدل NLP روی متن جدید

در زبانه **Run NLP model** امکان اجرای مدل روی یک نظر یا یک فایل CSV وجود دارد. فایل CSV باید ستون `reviewText` داشته باشد. ستون‌های `summary`، `reviewerID` و `asin` اختیاری هستند.

### محدودیت فایل‌های تحویلی تیم

آرشیو تیم شامل فایل‌های احتمال مدل‌ها (`classical_probs.npz` و `transformer_probs.npz`)، فایل‌های Submission و اسکریپت‌های آموزش است؛ اما فایل‌های خام `train_data.csv`، `test_data.csv` و `title_brand.csv` در آن نیستند. همچنین موارد زیر وجود ندارند:

- فایل خام `train_data.csv`
- مدل کلاسیک فیت‌شده به‌همراه Vectorizerها
- وزن‌های Transformer و فایل‌های Tokenizer برای اجرای مستقیم روی متن جدید

فایل‌های احتمال ذخیره‌شده فقط برای ارزیابی نمونه‌های قبلی هستند و نمی‌توان از آن‌ها برای پیش‌بینی متن جدید استفاده کرد.

### ساخت Checkpoint قابل اجرا برای شاخه کلاسیک

فایل اصلی آموزش را در مسیر زیر قرار دهید:

```text
sources/nlp_team/amazon-sentiment/data/train_data.csv
```

سپس از صفحه **NLP Dashboard & Run Model**، نمای **Run NLP model**، دکمه آموزش را اجرا کنید.

روش جایگزین از CMD:

```bat
.venv\Scripts\python.exe scripts\train_nlp_inference.py --train-csv sources\nlp_team\amazon-sentiment\data\train_data.csv --n-train 200000
```

خروجی در این مسیر ذخیره می‌شود:

```text
models/nlp_classical.joblib
```

پس از ساخت Checkpoint، امکان امتیازدهی یک نظر یا بارگذاری CSV فراهم می‌شود. این Runner شاخه **Classical TF-IDF + Logistic Regression** را با روش ویژگی‌های تاریخی Reviewer/Product اجرا می‌کند. آموزش آن یک خروجی قابل اجرا برای متن جدید می‌سازد؛ این Checkpoint، فایل وزن مدل اصلی تیم یا Ensemble نهایی نیست.

### وضعیت Transformer و Ensemble

در داشبورد اصلی، نتیجه‌های Classical، Transformer و Ensemble از احتمال‌های ذخیره‌شده گزارش شده‌اند. برای اجرای Ensemble روی متن جدید، وزن‌های دقیق Transformer، Tokenizer و شناسه مدل پایه باید در اختیار باشد. این موارد در ZIP تیم نبودند؛ بنابراین این نسخه خروجی زنده Transformer یا Ensemble را به‌صورت ساختگی نمایش نمی‌دهد.

گزینه EM label-shift در Runner فقط برای Batch حداقل ۱۰۰۰ ردیفی فعال است و به فرض ثبات توزیع شرطی کلاس‌ها وابسته است. برای یک نظر یا Batch کوچک، احتمال‌های کالیبره‌شده بدون این اصلاح نمایش داده می‌شوند.

## اجرای مدل‌های ویدیویی

برای اجرای واقعی مدل‌های Computer Vision، فایل‌های وزن صحیح باید در مسیرهای زیر قرار گیرند:

```text
models/detection.pth
models/anticipation.pth
```

دو فایل وزن در این بسته نیستند. معماری و مسیرهای دو مدل متفاوت‌اند؛ یک فایل وزن را در هر دو مسیر کپی نکنید. جزئیات در `models/README.txt` آمده است.

## ساختار منابع

- `assets/nlp_team_dashboard.html`: نسخه اصلی داشبورد تعاملی NLP که در صفحه Streamlit نمایش داده می‌شود.
- `sources/nlp_team/amazon-sentiment/`: کد، نوت‌بوک‌ها، گزارش فارسی، اسکریپت‌ها و خروجی‌های آرشیو اصلی تیم NLP.
- `sources/`: نوت‌بوک‌ها و لاگ‌های اصلی Computer Vision.
- `data/`: جدول‌های تحلیلی مورد نیاز داشبورد Computer Vision.
- `models/`: مسیر Checkpointهای محلی؛ وزن‌های خصوصی/حجیم در ZIP قرار نگرفته‌اند.
- `nlp_runtime.py` و `scripts/train_nlp_inference.py`: آموزش Checkpoint کلاسیک قابل اجرا و پیش‌بینی متن جدید.
- `docs/CrashLens_Final_v6_Team_Guide_FA.docx`: راهنمای Word برای ارسال به اعضای تیم.

## ترتیب پیشنهادی برای ارائه ۳۰ دقیقه‌ای

- **۰–۳ دقیقه:** معرفی هر دو مسئله در CV Overview و NLP Dashboard.
- **۳–۱۰ دقیقه:** معماری دو مدل CV و داده‌های Nexar.
- **۱۰–۱۶ دقیقه:** گراف آزمایش‌ها، پاک‌سازی داده و Data Leakage.
- **۱۶–۲۰ دقیقه:** نتایج و خطاهای Computer Vision.
- **۲۰–۲۶ دقیقه:** داشبورد NLP؛ Ratings، Words، Warranty و Model.
- **۲۶–۲۹ دقیقه:** اجرای یک نمونه NLP یا Video Demo، فقط اگر Checkpoint آماده باشد.
- **۲۹–۳۰ دقیقه:** جمع‌بندی و محدودیت‌ها.

## نصب وابستگی‌های اختیاری برای Video Inference

```bat
.venv\Scripts\python.exe -m pip install -r requirements-inference.txt --default-timeout=120 --retries=10
```

